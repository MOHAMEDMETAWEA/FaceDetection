import nbformat

def patch_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Patch FaceTracker class
            if 'class FaceTracker(Model):' in cell.source:
                cell.source = (
                    "class FaceTracker(Model): \n"
                    "    def __init__(self, model, **kwargs): \n"
                    "        super().__init__(**kwargs)\n"
                    "        self.model = model\n\n"
                    "    def compile(self, opt, classloss, localizationloss, **kwargs):\n"
                    "        super().compile(**kwargs)\n"
                    "        self.closs = classloss\n"
                    "        self.lloss = localizationloss\n"
                    "        self.optimizer = opt\n\n"
                    "    def train_step(self, batch):\n"
                    "        X, y = batch\n"
                    "        \n"
                    "        with tf.GradientTape() as tape:            \n"
                    "            classes, coords = self.model(X, training=True)\n"
                    "            \n"
                    "            # Ensure shapes are known to avoid rank errors\n"
                    "            classes = tf.ensure_shape(classes, [None, 1])\n"
                    "            coords = tf.ensure_shape(coords, [None, 4])\n"
                    "            y_class = tf.ensure_shape(y[0], [None, 1])\n"
                    "            y_coord = tf.ensure_shape(y[1], [None, 4])\n"
                    "            \n"
                    "            batch_classloss = self.closs(y_class, classes)\n"
                    "            mask = tf.cast(y_class, tf.float32)\n"
                    "            batch_localizationloss = self.lloss(tf.cast(y_coord, tf.float32) * mask, coords * mask)\n"
                    "            \n"
                    "            total_loss = batch_localizationloss + 0.5 * batch_classloss\n"
                    "            \n"
                    "        grad = tape.gradient(total_loss, self.model.trainable_variables)\n"
                    "        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))\n"
                    "        \n"
                    "        return {\"loss\": total_loss, \"class_loss\": batch_classloss, \"regress_loss\": batch_localizationloss}\n"
                    "    \n"
                    "    def test_step(self, batch): \n"
                    "        X, y = batch\n"
                    "        \n"
                    "        classes, coords = self.model(X, training=False)\n"
                    "        \n"
                    "        classes = tf.ensure_shape(classes, [None, 1])\n"
                    "        coords = tf.ensure_shape(coords, [None, 4])\n"
                    "        y_class = tf.ensure_shape(y[0], [None, 1])\n"
                    "        y_coord = tf.ensure_shape(y[1], [None, 4])\n"
                    "        \n"
                    "        batch_classloss = self.closs(y_class, classes)\n"
                    "        mask = tf.cast(y_class, tf.float32)\n"
                    "        batch_localizationloss = self.lloss(tf.cast(y_coord, tf.float32) * mask, coords * mask)\n"
                    "        \n"
                    "        total_loss = batch_localizationloss + 0.5 * batch_classloss\n"
                    "        \n"
                    "        return {\"loss\": total_loss, \"class_loss\": batch_classloss, \"regress_loss\": batch_localizationloss}\n"
                    "        \n"
                    "    def call(self, X, **kwargs): \n"
                    "        return self.model(X, **kwargs)"
                )
            
            # Patch optimizer setup to fix decay warning
            elif 'tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)' in cell.source or 'opt = tf.keras.optimizers.Adam(learning_rate=0.0001)' in cell.source:
                cell.source = "opt = tf.keras.optimizers.Adam(learning_rate=0.0001)"
            
            # Patch model.fit call
            elif 'hist = model.fit(' in cell.source:
                cell.source = (
                    "hist = model.fit(\n"
                    "    train,\n"
                    "    epochs=10,\n"
                    "    validation_data=val,\n"
                    "    callbacks=[tensorboard_callback]\n"
                    ")"
                )
            
            # Patch pip install lines
            if '%pip install' in cell.source:
                cell.source = cell.source.replace('%pip install labelme opencv-python matplotlib albumentations split-folders\\n', '%pip install labelme opencv-python matplotlib albumentations split-folders')
                cell.source = cell.source.replace('%pip install labelme opencv-python matplotlib albumentations split-folders\n', '%pip install labelme opencv-python matplotlib albumentations split-folders')

    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Patched {filepath}")

if __name__ == "__main__":
    patch_notebook('FaceDetection.ipynb')
