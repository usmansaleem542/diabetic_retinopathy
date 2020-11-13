import pandas as pd
import matplotlib.pyplot as plt
from PPImage import PPImage
import numpy as np
from PIL import Image
import os
import config

def plot_df_count(df, column='diagnosis'):
    df_plot = df[column].value_counts().sort_index()
    print(df_plot)
    df_plot.plot.bar(df_plot)
    plt.show()

def preprocess_image(row):
    image = PPImage()
    imgName = f'{row.id_code}.{config.IMAGE_EXTENSION}'
    image.from_zip(config.ZIP_FILE_PATH, imgName, config.FOLDER_IN_ZIP) #perfectExample

    if np.sum(image.data, axis=2).mean() < 15:
        return False

    target.data = target.resize(dim=(image.data.shape[1], image.data.shape[0]))
    image.hist_match_rgb(target=target.data)
    image.data = image.crop_image_only_outside(image.data, tol=15)

    image.data = image.hist_equalize(image.data)
    image.export(f'{config.HIST_MATCH_PATH}{imgName}', quality='good')

    orig = image.crop_image_only_outside(np.array(image.image), tol=15)
    orig = image.hist_equalize(orig)
    orig = Image.fromarray(orig)
    orig.save(f'{config.HIST_EQL_PATH}{imgName}', subsampling=0, quality=100)
    return True



# preprocess_image()

'''
idrid = pd.read_csv(root_path+'idrid.csv')
idrid['zip'] = 'idrid.zip'

for i in range(len(idrid)):
    row = idrid.iloc[i]
    print(i, "Processing: ", row.id_code)
    savePath = root_path + 'idrid/'
    preprocess_image(root_path, 'idrid/', row, '.jpg', savePath)


aptos = pd.read_csv(root_path+'aptos.csv')
aptos['zip'] = 'aptos.zip'
for i in range(len(aptos)):
    row = aptos.iloc[i]
    print(i, "Processing: ", row.id_code)
    savePath = root_path + 'aptos/'
    preprocess_image(root_path, '', row, '.png', savePath)

'''


target = PPImage(config.TARGET_IMAGE)

google = pd.read_csv(config.CSV_PATH)
os.makedirs(config.HIST_MATCH_PATH, exist_ok=True)
os.makedirs(config.HIST_EQL_PATH, exist_ok=True)

existing = os.listdir(config.HIST_MATCH_PATH)
existing2 = os.listdir(config.HIST_EQL_PATH)
errors = []
for i in range(len(google)):

    row = google.iloc[i]
    img = f'{row.id_code}.{config.IMAGE_EXTENSION}'
    if img in existing and img in existing2:
        continue
    print(i, "Processing: ", row.id_code)
    done = preprocess_image(row)
    if not done:
        errors.append(img)
        pd.DataFrame(errors, columns=['id_code']).to_csv("errors.csv", index=False)


# exit()
# all = pd.concat([idrid, google, aptos])
# print("==================== ALL ==================")
# plot_df_count(all)
# print("==================== APTOS ==================")
# plot_df_count(aptos)
# print("==================== GOOGLE ==================")
# plot_df_count(google)
# print("==================== IDRID ==================")
# plot_df_count(idrid)
#
# random_all = all.sample(frac=1).reset_index(drop=True)
# print("Done")

