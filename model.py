def set_seed(seed=42):
    import random, os
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)
set_seed(42)

def build_simple_cnn(input_shape=(224,224,3), num_classes=4):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def decode_and_resize(path, label, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def make_dataset(paths, labels, img_size=224, batch_size=16, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(lambda p,l: decode_and_resize(p,l,img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

class PerClassAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, class_names):
        super().__init__()
        self.val_ds = val_ds
        self.class_names = class_names
        self.history = {c: [] for c in class_names}
        self.cm_last = None
    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for x,y in self.val_ds:
            yp = self.model.predict(x, verbose=0)
            y_true.append(y.numpy()); y_pred.append(np.argmax(yp, axis=1))
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(self.class_names)))
        self.cm_last = cm
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_c = np.diag(cm) / cm.sum(axis=1).clip(min=1)
        for i,c in enumerate(self.class_names):
            self.history[c].append(float(acc_c[i]))

def infer_label_from_parts(parts):
    parts_lc = [p.lower() for p in parts]
    for p in reversed(parts_lc):
        for cls in CLASS_NAMES:
            if cls in p:
                return cls
    return None

def infer_subject_from_top(root, parts):
    root_parts = list(root.parts)
    for p in parts[len(root_parts):]:
        if SUBJECT_REGEX.match(p):
            return p
    if len(parts) > len(root_parts):
        return parts[len(root_parts)]
    return 'S0'

def path_allowed_by_keywords(path_str):
    if not INCLUDE_DIR_KEYWORDS:
        return True
    ps = path_str.lower()
    return any(k.lower() in ps for k in INCLUDE_DIR_KEYWORDS)

def scan_folder(DATA_ROOT):
    root = Path(DATA_ROOT).expanduser().resolve()
    paths, labels, subjects = [], [], []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if not path_allowed_by_keywords(str(p)):
                continue
            parts = list(p.parts)
            lab = infer_label_from_parts(parts)
            if lab is None:
                continue
            subj = infer_subject_from_top(root, parts)
            paths.append(str(p.resolve()))
            labels.append(lab)
            subjects.append(subj)
    return np.array(paths), np.array(labels), np.array(subjects)
