import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim
import pywt
import pandas as pd
from tqdm import tqdm
import shutil

gpus = tf.config.experimental.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ---------------------------
# Lightweight CNN denoiser

def build_cnn_denoiser(input_shape=(224, 224, 1)):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    out = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

class CNNDenoiser:
    def __init__(self, model=None, input_size=(224,224)):
        self.input_size = input_size
        self.model = model if model is not None else build_cnn_denoiser((*input_size,1))

    def train_on_synthetic(self, images_gray, epochs=5, batch_size=8):
        """images_gray: list/array of 2D uint8 grayscale images"""
        X, Y = [], []
        for im in images_gray:
            imr = cv2.resize(im, self.input_size)
            imn = imr.astype(np.float32) / 255.0
            noise = np.random.normal(0, 0.05, imn.shape).astype(np.float32)
            noisy = np.clip(imn + noise, 0.0, 1.0)
            X.append(noisy)
            Y.append(imn)
        if len(X)==0:
            return
        X = np.expand_dims(np.array(X), -1)
        Y = np.expand_dims(np.array(Y), -1)
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)
      
        self.model.save('lightweight_cnn_denoiser.h5')

    def denoise_image(self, image_2d):
 
        resized = cv2.resize(image_2d, self.input_size, interpolation=cv2.INTER_AREA)
        inp = resized.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=(0,-1))
        pred = self.model.predict(inp, verbose=0)
        pred = np.squeeze(pred)  # in 0..1
        out = (pred * 255.0).astype(np.uint8)
  
        return out

    def denoise_and_resize_back(self, image_2d, target_shape):
        """
        Denoise image (2D), but returns the denoised image resized to target_shape (h,w).
        """
        den = self.denoise_image(image_2d)
        den_back = cv2.resize(den, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        return den_back


def calculate_mse(orig, proc):
    return np.mean((orig.astype(np.float32) - proc.astype(np.float32)) ** 2)

def calculate_psnr(orig, proc):
    mse = calculate_mse(orig, proc)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(orig, proc):
    # ensure same size and single channel
    if orig.shape != proc.shape:
        proc = cv2.resize(proc, (orig.shape[1], orig.shape[0]))
    if len(orig.shape) == 3:
        origg = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    else:
        origg = orig
    if len(proc.shape) == 3:
        procg = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    else:
        procg = proc
    try:
        return ssim(origg, procg, data_range=255)
    except Exception:
        return 0.0

# ---------------------------
# Dual-path denoiser


class OCTDenoiserPipeline:
    def __init__(self, input_base, processed_base=None, classes=None, model_path=None,
                 cnn_input_size=(224,224), swt_level=1, fusion_alpha=0.6):
       
        self.input_base = input_base
        self.processed_base = processed_base if processed_base else os.path.join(input_base, 'processed')
        os.makedirs(self.processed_base, exist_ok=True)
        self.classes = classes
        if self.classes is None:
           
            self.classes = [d for d in sorted(os.listdir(input_base)) if os.path.isdir(os.path.join(input_base, d))]
        self.cnn_input_size = cnn_input_size
        self.swt_level = swt_level
        self.alpha = fusion_alpha

   
        self.metrics = []

        # CNN denoiser
        if model_path and os.path.exists(model_path):
            self.cnn = CNNDenoiser(build_cnn_denoiser((*cnn_input_size,1)), cnn_input_size)
            self.cnn.model = tf.keras.models.load_model(model_path)
        else:
            self.cnn = CNNDenoiser(input_size=cnn_input_size)


    @staticmethod
    def _ensure_even_pad(image_2d):
        """Pad by repeating last row/col so height and width are even (SWT requirement)"""
        h, w = image_2d.shape
        pad_h = 1 if (h % 2 != 0) else 0
        pad_w = 1 if (w % 2 != 0) else 0
        if pad_h == 0 and pad_w == 0:
            return image_2d, (0,0)
        pad_vals = ((0,pad_h), (0,pad_w))
        padded = np.pad(image_2d, pad_vals, mode='edge')
        return padded, (pad_h, pad_w)

    @staticmethod
    def _remove_padding(image_2d, pad):
        pad_h, pad_w = pad
        if pad_h == 0 and pad_w == 0:
            return image_2d
        h = image_2d.shape[0] - pad_h
        w = image_2d.shape[1] - pad_w
        return image_2d[:h, :w]

    def _denoise_swt_coeff(self, coeff):
     
        # store range
        cmin, cmax = np.min(coeff), np.max(coeff)
        if cmax - cmin < 1e-8:
          
            return coeff.copy()
       
        scaled = (coeff - cmin) / (cmax - cmin)
        scaled_u8 = (scaled * 255.0).astype(np.uint8)
 
        denoised_u8 = self.cnn.denoise_and_resize_back(scaled_u8, coeff.shape)
    
        denoised_scaled = denoised_u8.astype(np.float32) / 255.0
     
        denoised_orig_scale = denoised_scaled * (cmax - cmin) + cmin
        return denoised_orig_scale

    def swt_path(self, image_2d):
     
        padded, pad = self._ensure_even_pad(image_2d)
       
        arr = padded.astype(np.float32)
     
        coeffs = pywt.swt2(arr, 'db4', level=self.swt_level)

        denoised_coeffs = []
        for (cA, (cH, cV, cD)) in coeffs:
            
            cA_d = self._denoise_swt_coeff(cA)
            cH_d = self._denoise_swt_coeff(cH)
            cV_d = self._denoise_swt_coeff(cV)
            cD_d = self._denoise_swt_coeff(cD)
            denoised_coeffs.append((cA_d, (cH_d, cV_d, cD_d)))
       
        rec = pywt.iswt2(denoised_coeffs, 'db4')
       
        rec_clipped = np.clip(rec, 0, 255).astype(np.uint8)
    
        rec_unpad = self._remove_padding(rec_clipped, pad)
        return rec_unpad

    # spatial CNN
    # ---------------------------
    def cnn_path(self, image_2d):
        
        return self.cnn.denoise_and_resize_back(image_2d, image_2d.shape)

    # ---------------------------
    # Fusion
  
    def fuse(self, spatial_uint8, frequency_uint8, alpha=None):
        a = self.alpha if alpha is None else alpha
    
        if spatial_uint8.shape != frequency_uint8.shape:
            frequency_uint8 = cv2.resize(frequency_uint8, (spatial_uint8.shape[1], spatial_uint8.shape[0]),
                                         interpolation=cv2.INTER_AREA)
        fused = (a * spatial_uint8.astype(np.float32) + (1.0 - a) * frequency_uint8.astype(np.float32))
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        return fused


    def process_single_image(self, image_path, out_path):
        try:
            im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if im is None:
                print(f"Could not read: {image_path}")
                return None
            # Path A
            spatial = self.cnn_path(im)
            # Path B
            frequency = self.swt_path(im)
            # Fuse
            fused = self.fuse(spatial, frequency)
    
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
            cv2.imwrite(out_path, fused)
         
            mse_v = calculate_mse(im, fused)
            psnr_v = calculate_psnr(im, fused)
            ssim_v = calculate_ssim(im, fused)
            return {'image': os.path.basename(image_path),
                    'mse': mse_v, 'psnr': psnr_v, 'ssim': ssim_v}
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None


    def process_dataset(self, retrain_on_samples=15, epochs=5, save_metrics_csv=True):
   
        sample_images = []
        for cls in self.classes:
            cls_path = os.path.join(self.input_base, cls)
            if not os.path.isdir(cls_path):
                continue
            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff'))]
            for f in files[:retrain_on_samples]:
                p = os.path.join(cls_path, f)
                im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    sample_images.append(im)
        if len(sample_images) > 0:
            try:
                self.cnn.train_on_synthetic(sample_images, epochs=epochs)
            except Exception as e:
                print(f"Quick training failed: {e}")

        if os.path.exists(self.processed_base):
        
            pass

        metrics_list = []
        for cls in self.classes:
            cls_in = os.path.join(self.input_base, cls)
            cls_out = os.path.join(self.processed_base, cls)
            if not os.path.isdir(cls_in):
                continue
            os.makedirs(cls_out, exist_ok=True)
            files = [f for f in sorted(os.listdir(cls_in)) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff'))]
            print(f"\nProcessing class: {cls}  (count: {len(files)})")
            for fname in tqdm(files):
                p_in = os.path.join(cls_in, fname)
                p_out = os.path.join(cls_out, fname)
                res = self.process_single_image(p_in, p_out)
                if res is not None:
                    res['class'] = cls
                    metrics_list.append(res)

        if save_metrics_csv:
            if len(metrics_list) > 0:
                df = pd.DataFrame(metrics_list)[['image','class','mse','psnr','ssim']]
                csv_path = os.path.join(self.processed_base, 'denoising_metrics.csv')
                df.to_csv(csv_path, index=False)
       
                print("\n=== DENOISING METRICS SUMMARY (per class mean) ===")
                print(df.groupby('class')[['mse','psnr','ssim']].mean())
                print(f"\nMetrics saved to: {csv_path}")
            else:
                print("No.")

        return metrics_list


if __name__ == "__main__":


    CLASS_LIST = ['CSR','DR','MH','NORMAL']

    pipeline = OCTDenoiserPipeline(
        input_base=INPUT_BASE,
        processed_base=PROCESSED_BASE,
        classes=CLASS_LIST,
        model_path=None,         
        cnn_input_size=(224,224),
        swt_level=1,
        fusion_alpha=0.6
    )

    pipeline.process_dataset(retrain_on_samples=50, epochs=20, save_metrics_csv=True)

