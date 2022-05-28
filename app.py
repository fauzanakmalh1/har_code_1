import numpy as np
import pandas as pd
import itertools
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

FPS = 25
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
CLASSES_DICT = {0: 'No Cheat', 1: 'Read Text',
                2: 'Ask Friend', 5: 'Call Friend'}
CLASSES_LIST = ['No Cheat', 'Read Text', 'Ask Friend', 'Call Friend']
MODEL_OUTPUT_SIZE = len(CLASSES_LIST)
BASE_DIR = './model'

st.set_page_config(page_title='Human Activity Recognition')
st.set_option('deprecation.showfileUploaderEncoding', False)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css('./style/style.css')

st.sidebar.write(
    'Human Activity Recognition Berdasarkan Webcam Menggunakan Metode MobileNet')

menu_name = st.sidebar.selectbox(
    'Pilih Menu', ['Halaman Depan', 'Simulasi Pelatihan Model', 'Prediksi Gambar'])

def image_classification(filename):
    model = load_model(f'{BASE_DIR}/checkpoint/HAR_MobileNetV2_Model_Best.h5', compile=False)
    size = (IMAGE_HEIGHT, IMAGE_WIDTH)
    img = ImageOps.fit(filename, size, Image.ANTIALIAS)
    img = img_to_array(img)
    img = img.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    img = img.astype('float32')
    img = img / 255.0
    prediction = model.predict(img)
    prediction = prediction[0]
    predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(prediction)[
        ::-1]

    st.write('#### Prediksi Kelas dan Probabilitas-nya pada Gambar:')
    for predicted_label in predicted_labels_probabilities_averaged_sorted_indexes:
        predicted_class_name = CLASSES_LIST[predicted_label]
        predicted_probability = prediction[predicted_label]
        st.write(
            f'{predicted_class_name}: {(predicted_probability * 100):.2f}%')

def get_model_name(fold_var_new):
    return f'{BASE_DIR}/checkpoint/HAR_MobileNetV2_Model_fold-{str(fold_var_new)}.h5'


def get_model(fold_var_new, dense_layer_new, init_lr_new, epochs_new):
    st.write('---')
    st.write(f'#### [INFO] Membangun Model Fold-{str(fold_var_new)}')
    baseModel = MobileNetV2(weights='imagenet',
                            include_top=False,
                            input_tensor=Input(
                                shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
                            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                            classes=MODEL_OUTPUT_SIZE)
    baseModel.trainable = False

    headModel = baseModel.output
    headModel = Conv2D(100, (3, 3), activation='relu', input_shape=(
        IMAGE_HEIGHT, IMAGE_WIDTH, 3))(headModel)
    headModel = MaxPooling2D(pool_size=(2, 2))(headModel)
    headModel = Flatten(name='flatten')(headModel)
    if dense_layer_new == 1:
        headModel = Dense(512, activation='relu',
                          name='dense_layer_1')(headModel)
    if dense_layer_new == 3:
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_1')(headModel)
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_2')(headModel)
        headModel = Dense(512, activation='relu',
                          name='dense_layer_3')(headModel)
    if dense_layer_new == 5:
        headModel = Dense(2048, activation='relu',
                          name='dense_layer_1')(headModel)
        headModel = Dense(2048, activation='relu',
                          name='dense_layer_2')(headModel)
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_3')(headModel)
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_4')(headModel)
        headModel = Dense(512, activation='relu',
                          name='dense_layer_5')(headModel)
    if dense_layer_new == 7:
        headModel = Dense(4096, activation='relu',
                          name='dense_layer_1')(headModel)
        headModel = Dense(4096, activation='relu',
                          name='dense_layer_2')(headModel)
        headModel = Dense(2048, activation='relu',
                          name='dense_layer_3')(headModel)
        headModel = Dense(2048, activation='relu',
                          name='dense_layer_4')(headModel)
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_5')(headModel)
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_6')(headModel)
        headModel = Dense(512, activation='relu',
                          name='dense_layer_7')(headModel)
    headModel = Dense(MODEL_OUTPUT_SIZE, activation='softmax',
                      name='dense_layer_out')(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False

    opt = Adam(learning_rate=init_lr_new, decay=init_lr_new / epochs_new)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model


def plot_history(H, fold_var_new):
    st.write('---')
    st.write(f'#### [INFO] Plot Fold-{str(fold_var_new)}')
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(range(1, len(H.history['loss'])+1),
             H.history['loss'], label='train_loss')
    plt.plot(range(1, len(H.history['val_loss'])+1),
             H.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    savefig_dir = f'{BASE_DIR}/plot/plot_loss_fold-{str(fold_var_new)}.png'
    plt.savefig(savefig_dir, bbox_inches='tight')
    image = Image.open(savefig_dir)
    st.write('Plot Pelatihan dan Validasi Loss')
    st.image(image, use_column_width=True)

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(range(1, len(H.history['accuracy'])+1),
             H.history['accuracy'], label='train_acc')
    plt.plot(range(1, len(H.history['val_accuracy'])+1),
             H.history['val_accuracy'], label='val_acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    savefig_dir = f'{BASE_DIR}/plot/plot_accuracy_fold-{str(fold_var_new)}.png'
    plt.savefig(savefig_dir, bbox_inches='tight')
    image = Image.open(savefig_dir)
    st.write('Plot Pelatihan dan Validasi Akurasi')
    st.image(image, use_column_width=True)

def plot_confusion_matrix(cm, classes, fold_var_new, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues):
    st.write('---')
    st.write(f'#### [INFO] Confusion Matrix Fold-{str(fold_var_new)}')
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        st.write(f'Normalized Confusion Matrix')
    else:
        st.write(f'Confusion Matrix, Without Normalization')

    thresh = cm.max()/2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if normalize:
        savefig_dir = f'{BASE_DIR}/plot/plot_confusion_matrix_normalized_fold-{str(fold_var_new)}.png'
    else:
        savefig_dir = f'{BASE_DIR}/plot/plot_confusion_matrix_fold-{str(fold_var_new)}.png'
    plt.savefig(savefig_dir, bbox_inches='tight')
    image = Image.open(savefig_dir)
    st.image(image, use_column_width=True)


if menu_name == 'Halaman Depan':
    st.write('## Halaman Depan')
    st.write('---')
    st.write('#### Judul Skripsi:')
    st.write(
        '#### Human Activity Recognition Berdasarkan Webcam Menggunakan Metode MobileNet')
    st.write('---')
    st.write('#### Abstrak:')
    st.markdown('<div style="text-align: justify;"><p>&nbsp;&nbsp;&nbsp;&nbsp;Manusia tidak bisa terlepas dari aktivitas sehari-hari yang mana merupakan bagian dari aktivitas kehidupan manusia. Human Activity Recognition (HAR) atau pengenalan aktivitas manusia saat ini merupakan salah satu topik yang sedang banyak diteliti seiring dengan pesatnya kemajuan di bidang teknologi yang berkembang saat ini. Hampir semua bidang terdampak dari pandemi COVID-19 yang mempengaruhi aktivitas manusia sehingga menjadi lebih terbatas. Salah satu bidang yang paling terdampak yaitu pendidikan, di mana kampus menerapkan sistem pembelajaran daring, sehingga dosen lebih sulit untuk mengawasi pembelajaran maupun ujian yang dilakukan secara daring karena tidak dapat mengawasi aktivitas yang dilakukan mahasiswa secara langsung.</p></div>', unsafe_allow_html=True)

if menu_name == 'Simulasi Pelatihan Model':
    st.write('## Simulasi Pelatihan Model')
    st.write('---')
    st.warning('Catatan:\nData yang digunakan pada simulasi pelatihan dan pengujian model pada halaman ini menggunakan 10% dari data yang digunakan pada penelitian.')
    st.write('---')
    st.write('#### Atur Hyperparameter')
    init_lr_new = st.select_slider(
        'Learning Rate', options=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5], value=1e-1)
    epochs_new = st.slider('Epochs', min_value=2, max_value=40, value=2, step=1)
    early_stopping_new = st.slider('Early Stopping Patience', min_value=1,
                                   max_value=40, value=10, step=1)
    batch_size_new = st.select_slider(
        'Batch Size', options=[8, 16, 32, 64, 128, 256], value=8)
    dense_layer_new = st.select_slider(
        'Dense Layer', options=[1, 3, 5, 7], value=1)
    st.write('---')
    st.write(f'''\n#### Cek Hyperparameter yang Diatur
        \nLearning Rate: {init_lr_new}
        \nEpochs: {epochs_new}
        \nEarly Stopping: {early_stopping_new}
        \nBatch Size: {batch_size_new}
        \nDense Layer: {dense_layer_new}''', unsafe_allow_html=True)
    if dense_layer_new == 1:
        st.code(
            f'''Dense(512, activation='relu', name='dense_layer_1')''', language='python')
    if dense_layer_new == 3:
        st.code(f'''Dense(1024, activation='relu', name='dense_layer_1')\nDense(1024, activation='relu', name='dense_layer_2')\nDense(512, activation='relu', name='dense_layer_3')''', language='python')
    if dense_layer_new == 5:
        st.code(f'''Dense(2048, activation='relu', name='dense_layer_1')\nDense(2048, activation='relu', name='dense_layer_2')\nDense(1024, activation='relu', name='dense_layer_3')\nDense(1024, activation='relu', name='dense_layer_4')\nDense(512, activation='relu', name='dense_layer_5')''', language='python')
    if dense_layer_new == 7:
        st.code(f'''Dense(4096, activation='relu', name='dense_layer_1')\nDense(4096, activation='relu', name='dense_layer_2')\nDense(2048, activation='relu', name='dense_layer_3')\nDense(2048, activation='relu', name='dense_layer_4')\nDense(1024, activation='relu', name='dense_layer_5')\nDense(1024, activation='relu', name='dense_layer_6')\nDense(512, activation='relu', name='dense_layer_7')''', language='python')

    if st.button('Jalankan Model'):
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=[0.9, 1.0],
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1./255)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_data = pd.read_csv(
            f'{BASE_DIR}/data/split_perdata/train_labels.csv')
        test_data = pd.read_csv(
            f'{BASE_DIR}/data/split_perdata/test_labels.csv')

        train_y = train_data.label
        train_x = train_data.drop(['label'], axis=1)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=47)
        data_kfold = pd.DataFrame()

        validation_accuracy = []
        validation_loss = []
        fold_var_new = 1

        for train_index, val_index in list(skf.split(train_x, train_y)):
            training_data = train_data.iloc[train_index]
            validation_data = train_data.iloc[val_index]

            train_data_generator = train_datagen.flow_from_dataframe(
                training_data,
                directory=f'{BASE_DIR}/data/split_perdata/train/',
                x_col='filename',
                y_col='label',
                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                color_mode='rgb',
                class_mode='categorical',
                batch_size=batch_size_new,
                shuffle=True)
            valid_data_generator = train_datagen.flow_from_dataframe(
                validation_data,
                directory=f'{BASE_DIR}/data/split_perdata/train/',
                x_col='filename',
                y_col='label',
                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                color_mode='rgb',
                class_mode='categorical',
                batch_size=batch_size_new,
                shuffle=True)

            model = get_model(fold_var_new, dense_layer_new,
                              init_lr_new, epochs_new)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                get_model_name(fold_var_new),
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='max')
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=early_stopping_new,
                verbose=1,
                mode='auto',
                baseline=None)

            placeholder = st.empty()
            placeholder.write('Pelatihan dan Pengujian Model Sedang Diproses')
            with st.spinner('Silahkan Tunggu ...'):
                history = model.fit(train_data_generator,
                                    steps_per_epoch=train_data_generator.samples // train_data_generator.batch_size,
                                    epochs=epochs_new,
                                    validation_data=valid_data_generator,
                                    validation_steps=valid_data_generator.samples // valid_data_generator.batch_size,
                                    verbose=1,
                                    callbacks=[checkpoint, early_stopping])
                plot_history(history, fold_var_new)
            placeholder.empty()

            model.load_weights(
                f'{BASE_DIR}/checkpoint/HAR_MobileNetV2_Model_fold-{str(fold_var_new)}.h5')

            test_data_generator = test_datagen.flow_from_dataframe(
                test_data,
                directory=f'{BASE_DIR}/data/split_perdata/test/',
                x_col='filename',
                y_col='label',
                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                color_mode='rgb',
                class_mode='categorical',
                batch_size=batch_size_new,
                shuffle=False)
            test_data_generator.reset()
            test_steps_per_epoch = np.math.ceil(
                test_data_generator.samples / test_data_generator.batch_size)
            predictions = model.predict_generator(
                test_data_generator, steps=test_steps_per_epoch)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = test_data_generator.classes
            class_labels = list(test_data_generator.class_indices.keys())

            cm = confusion_matrix(true_classes, predicted_classes)
            plot_confusion_matrix(
                cm, class_labels, fold_var_new, normalize=True)

            st.write('---')
            st.write(
                f'#### [INFO] Classification Report Fold-{str(fold_var_new)}')
            report = classification_report(
                true_classes, predicted_classes, target_names=class_labels, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            data_kfold[fold_var_new] = predicted_classes
            st.write('---')
            st.write(f'#### [INFO] Fold {fold_var_new} Selesai Dijalankan')
            if fold_var_new == 3:
                st.write('---')
                st.success('Pelatihan dan Pengujian Model Selesai Dijalankan')
            tf.keras.backend.clear_session()
            fold_var_new += 1

if menu_name == 'Prediksi Gambar':
    st.write('## Prediksi Gambar')
    st.write('---')
    uploaded_image = st.file_uploader(
        'Silahkan Pilih Gambar yang Ingin Diprediksi', type=['jpg', 'png'])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.write('---')
        st.write('#### Gambar yang Diupload:')
        st.image(image, use_column_width=True)
        st.write('---')
        placeholder = st.empty()
        placeholder.write('Prediksi Sedang Diproses')
        image_classification(image)
        placeholder.empty()