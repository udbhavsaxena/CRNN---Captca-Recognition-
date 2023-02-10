import os
import glob
import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import engine
from model import CaptchaModel

def decode_predictions(preds, encoder):
    preds = preds.permute(1,0,2) # bs,ts,preds
    preds = torch.softmax(preds,2)
    preds = torch.argmax(preds,2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []

    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j,:]:
            k -= 1
            if k == -1:
                temp.append('*')
            else:
                temp.append(encoder.inverse_transform(k)[0])

        tp = ''.join(temp)
        cap_preds.append(tp)

    return cap_preds


def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, '*.png'))
    targets_orig = [x.split('/')[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_encoded = [lbl_enc.transform(x) for x in targets]
    targets_encoded = np.array(targets_encoded) + 1

    # print(image_files)
    print(targets_encoded)
    print(len(lbl_enc.classes_))

    train_images, test_images, train_targets, test_targets, train_orig_targets, test_orig_targets = model_selection.train_test_split(
        image_files, targets_encoded, targets_orig, test_size=0.1, random_state=42)

    train_dataset = dataset.ClassificationDataset(
        image_path=train_images, targets=train_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        shuffle = True
    )

    test_dataset = dataset.ClassificationDataset(
        image_path=test_images, targets=test_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        shuffle = False
    )

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)

    optmizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optmizer, factor=0.8, patience=5, verbose=True
    )
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_function(model, train_loader, optmizer)
        val_preds, val_loss = engine.eval_function(model,train_loader)
        valid_cap_preds = []
        for vp in val_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_cap_preds.extend(current_preds)
        print(list(zip(test_orig_targets, valid_cap_preds))[6:11])
        print(f'Epoch: {epoch}, train_loss: {train_loss}, valid_loss:{val_loss}')

    # defining the model
if __name__ == '__main__':
    run_training()
    #75 Values: '6666'''''ddddd''''d778h''''''
    #how to decode preds
    #refer to decode pred
