import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
import os

"""
for i in os.listdir('train/'): #train 파일명 출력하여 개, 고양이 분류
    if 'cat' in i:
        shutil.copyfile('train/' + i, 'dataset/cat/' + i)
    if 'dog' in i:
        shutil.copyfile('train/' + i, 'dataset/dog/' + i)
"""

#((xxxx), (yyyy)) 왼쪽 이미지를 숫자화 한것이 64개, 정답들 개인지 고양이인지
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    image_size=(64, 64),
    batch_size=64,
    subset='training', #training 데이터 셋
    validation_split=0.2, #데이터를 0.2개로 쪼개겠다
    seed=1234
) #데이터중 80%

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    image_size=(64, 64),
    batch_size=64,
    subset='validation', #validation 데이터 셋
    validation_split=0.2, #데이터를 0.2개로 쪼개겠다
    seed=1234
) #데이터중 20%

def pre_processing(i, answer):
    i = tf.cast(i / 255.0, tf.float32)
    return i, answer
#전처리를 통해 모든 요소를 돌면서 이미지 값을 0 ~ 255 -> 0 ~ 1로 만들어줌, 연산 속도 빠르게 하기 위함
train_ds = train_ds.map(pre_processing)
val_ds = val_ds.map(pre_processing)

'''
#데이터 출력
for i, answer in train_ds.take(1):
    print(i)
    print(answer)
    plt.imshow(i[0].numpy().astype(uint8))
    plt.show()
'''

model = tf.keras.Sequential([
    #이미지 증강, 사진 뒤집기 , input_shape은 첫레이어에
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(64, 64, 3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),

    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"), #칼라 사진
    tf.keras.layers.MaxPool2D( (2, 2) ),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),  # 칼라 사진
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.2), #오버 피팅 완화, 윗 레이어의 노드를 일부제거 20%
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),  # 칼라 사진
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid"),
]) #binary_crossentropy에서는 마지막 레이어에 sigmoid가 필요하다

model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=15)
