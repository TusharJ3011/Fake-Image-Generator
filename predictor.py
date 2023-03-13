from tensorflow import keras
import numpy as np
import cv2

generator = keras.models.load_model('rightHandFakeGeneratorFinal.h5')

num_images = 10000

noise = np.random.normal(0, 1, (num_images, 100))
gen_imgs = generator.predict(noise)
for i, gen_img in enumerate(gen_imgs):
    gen_img = 0.5 * gen_img + 0.5
    gen_img = gen_img * 255
    gen_img = gen_img.astype(np.uint8)
    gen_img = np.squeeze(gen_img, axis=-1)
    cv2.imwrite(f"images/Noised Fake/img_{i+1}.png", gen_img)
