# # CNN Cat vs Dog Classification - Complete Template
# # ================================================================
# # Template untuk klasifikasi gambar kucing dan anjing menggunakan CNN
# # Memenuhi semua kriteria: Sequential model, Conv2D, Pooling, akurasi >85%
# # ================================================================

# # Import Library yang Diperlukan
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam, RMSprop, SGD
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix
# import os
# import time

# print(f"TensorFlow version: {tf._version_}")

# # ================================================================
# # 1. DATA PREPARATION & SPLITTING
# # ================================================================

# # Path ke dataset
# base_dir = '/content/dataset/combined'
# train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation') 
# test_dir = os.path.join(base_dir, 'test')

# # Membuat direktori jika belum ada
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(validation_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# # Data splitting - 70% train, 20% validation, 10% test
# from sklearn.model_selection import train_test_split
# import shutil

# def split_data(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.2):
#     """
#     Split data into train, validation, and test sets
#     """
#     categories = ['cats', 'dogs']
    
#     for category in categories:
#         # Create category directories
#         os.makedirs(os.path.join(train_dir, category), exist_ok=True)
#         os.makedirs(os.path.join(val_dir, category), exist_ok=True)
#         os.makedirs(os.path.join(test_dir, category), exist_ok=True)
        
#         # Get all files for this category
#         category_dir = os.path.join(source_dir, category)
#         if os.path.exists(category_dir):
#             files = os.listdir(category_dir)
#             files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
#             # Split files
#             train_files, temp_files = train_test_split(files, train_size=train_ratio, random_state=42)
#             val_files, test_files = train_test_split(temp_files, train_size=val_ratio/(1-train_ratio), random_state=42)
            
#             # Copy files to respective directories
#             for file in train_files:
#                 shutil.copy2(os.path.join(category_dir, file), os.path.join(train_dir, category, file))
#             for file in val_files:
#                 shutil.copy2(os.path.join(category_dir, file), os.path.join(val_dir, category, file))
#             for file in test_files:
#                 shutil.copy2(os.path.join(category_dir, file), os.path.join(test_dir, category, file))
            
#             print(f"{category}: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")

# # Uncomment untuk melakukan split data (jika belum di-split)
# # split_data('/content/dataset/combined', train_dir, validation_dir, test_dir)

# # ================================================================
# # 2. IMAGE DATA GENERATOR
# # ================================================================

# # Data augmentation untuk training set
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     zoom_range=0.2,
#     shear_range=0.2,
#     fill_mode='nearest'
# )

# # Hanya rescaling untuk validation dan test set
# validation_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# # Data generators
# IMG_HEIGHT, IMG_WIDTH = 150, 150
# BATCH_SIZE = 32

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary'
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     shuffle=False
# )

# print("Data generators created successfully!")
# print(f"Training samples: {train_generator.samples}")
# print(f"Validation samples: {validation_generator.samples}")
# print(f"Test samples: {test_generator.samples}")

# # ================================================================
# # 3. MODEL EXPERIMENTS
# # ================================================================

# def create_model_exp1():
#     """Model Exp 1 - CNN Architecture (Using 32 Neurons in Conv Layer)"""
#     tf.keras.backend.clear_session()
    
#     model = Sequential([
#         Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(32, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(32, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def create_model_exp2():
#     """Model Exp 2 - CNN Architecture Using 64 Neurons in Conv Layer"""
#     tf.keras.backend.clear_session()
    
#     model = Sequential([
#         Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(64, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(64, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def create_model_exp3():
#     """Model Exp 3 - CNN Architecture Using 128 Neurons in Conv Layer"""
#     tf.keras.backend.clear_session()
    
#     model = Sequential([
#         Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(128, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(128, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Flatten(),
#         Dense(256, activation='relu'),
#         Dropout(0.5),
#         Dense(128, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def create_model_exp4():
#     """Model Exp 4 - CNN Architecture (Using Learning Rate 0.001)"""
#     tf.keras.backend.clear_session()
    
#     model = Sequential([
#         Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(64, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(128, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def create_model_exp5():
#     """Model Exp 5 - CNN Architecture (Using Learning Rate 0.0001)"""
#     tf.keras.backend.clear_session()
    
#     model = Sequential([
#         Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(64, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(128, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer=Adam(learning_rate=0.0001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def create_model_exp6():
#     """Model Exp 6 - CNN Architecture (Using Learning Rate 0.00001)"""
#     tf.keras.backend.clear_session()
    
#     model = Sequential([
#         Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(64, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(128, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer=Adam(learning_rate=0.00001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def create_model_exp7():
#     """Model Exp 7 - CNN Architecture (Using Adam as Optimizer)"""
#     tf.keras.backend.clear_session()
    
#     model = Sequential([
#         Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(128, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(256, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Flatten(),
#         Dense(256, activation='relu'),
#         Dropout(0.5),
#         Dense(128, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer=Adam(),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def create_model_exp8():
#     """Model Exp 8 - CNN Architecture (Using SGD as Optimizer)"""
#     tf.keras.backend.clear_session()
    
#     model = Sequential([
#         Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(128, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Conv2D(256, (3, 3), padding='same', activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         Flatten(),
#         Dense(256, activation='relu'),
#         Dropout(0.5),
#         Dense(128, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# # ================================================================
# # 4. TRAINING AND EVALUATION FUNCTIONS
# # ================================================================

# def train_and_evaluate_model(model, model_name, epochs=25):
#     """
#     Train and evaluate a model
#     """
#     print(f"\n{'='*60}")
#     print(f"Training {model_name}")
#     print(f"{'='*60}")
    
#     # Model summary
#     print(model.summary())
    
#     # Calculate class weights for imbalanced dataset
#     total_train = train_generator.samples
#     weight_for_0 = (1 / train_generator.class_indices['cats']) * (total_train) / 2.0
#     weight_for_1 = (1 / train_generator.class_indices['dogs']) * (total_train) / 2.0
#     class_weight = {0: weight_for_0, 1: weight_for_1}
    
#     # Training
#     start_time = time.time()
#     history = model.fit(
#         train_generator,
#         steps_per_epoch=train_generator.samples // BATCH_SIZE,
#         epochs=epochs,
#         validation_data=validation_generator,
#         validation_steps=validation_generator.samples // BATCH_SIZE,
#         class_weight=class_weight,
#         verbose=1
#     )
#     training_time = time.time() - start_time
    
#     print(f"Training completed in {training_time:.2f} seconds")
    
#     # Plot training history
#     plot_training_history(history, model_name)
    
#     # Evaluate on test set
#     test_generator.reset()
#     test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    
#     # Make predictions
#     test_generator.reset()
#     predictions = model.predict(test_generator, verbose=0)
#     predicted_classes = (predictions > 0.5).astype(int).flatten()
    
#     # Confusion matrix and classification report
#     true_classes = test_generator.classes
    
#     print(f"\n{model_name} Results:")
#     print(f"Test Accuracy: {test_accuracy:.4f}")
#     print(f"Test Loss: {test_loss:.4f}")
    
#     # Confusion Matrix
#     cm = confusion_matrix(true_classes, predicted_classes)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
#     plt.title(f'{model_name} - Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.show()
    
#     # Classification Report
#     print("\nClassification Report:")
#     print(classification_report(true_classes, predicted_classes, 
#                               target_names=['Cat', 'Dog'], digits=4))
    
#     return history, test_accuracy, test_loss

# def plot_training_history(history, model_name):
#     """
#     Plot training and validation accuracy and loss
#     """
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
    
#     epochs_range = range(len(acc))
    
#     plt.figure(figsize=(15, 5))
    
#     # Plot accuracy
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, acc, 'r', label='Training Accuracy')
#     plt.plot(epochs_range, val_acc, 'b', label='Validation Accuracy')
#     plt.title(f'{model_name} - Training and Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True)
    
#     # Plot loss
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, loss, 'r', label='Training Loss')
#     plt.plot(epochs_range, val_loss, 'b', label='Validation Loss')
#     plt.title(f'{model_name} - Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()

# # ================================================================
# # 5. MODEL SAVING FUNCTIONS
# # ================================================================

# def save_model_all_formats(model, model_name):
#     """
#     Save model in SavedModel, TF-Lite, and TFJS formats
#     """
#     base_path = f'/content/saved_models/{model_name}'
#     os.makedirs(base_path, exist_ok=True)
    
#     # 1. SavedModel format
#     savedmodel_path = os.path.join(base_path, 'savedmodel')
#     model.save(savedmodel_path)
#     print(f"✅ SavedModel saved to: {savedmodel_path}")
    
#     # 2. TF-Lite format
#     converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
#     tflite_model = converter.convert()
    
#     tflite_path = os.path.join(base_path, f'{model_name}.tflite')
#     with open(tflite_path, 'wb') as f:
#         f.write(tflite_model)
#     print(f"✅ TF-Lite model saved to: {tflite_path}")
    
#     # 3. TFJS format
#     tfjs_path = os.path.join(base_path, 'tfjs')
#     os.system(f'tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model {savedmodel_path} {tfjs_path}')
#     print(f"✅ TFJS model saved to: {tfjs_path}")

# # ================================================================
# # 6. MAIN EXECUTION
# # ================================================================

# # Dictionary to store all results
# results = {}

# # List of model creation functions and their names
# model_configs = [
#     (create_model_exp1, "Model Exp 1 - CNN Architecture (Using 32 Neurons in Conv Layer)"),
#     (create_model_exp2, "Model Exp 2 - CNN Architecture Using 64 Neurons in Conv Layer"),
#     (create_model_exp3, "Model Exp 3 - CNN Architecture Using 128 Neurons in Conv Layer"),
#     (create_model_exp4, "Model Exp 4 - CNN Architecture (Using Learning Rate 0.001)"),
#     (create_model_exp5, "Model Exp 5 - CNN Architecture (Using Learning Rate 0.0001)"),
#     (create_model_exp6, "Model Exp 6 - CNN Architecture (Using Learning Rate 0.00001)"),
#     (create_model_exp7, "Model Exp 7 - CNN Architecture (Using Adam as Optimizer)"),
#     (create_model_exp8, "Model Exp 8 - CNN Architecture (Using SGD as Optimizer)")
# ]

# # ================================================================
# # EXECUTE INDIVIDUAL MODELS
# # ================================================================

# # Uncomment the model you want to train:

# # # Model Experiment 1
# # model_1 = create_model_exp1()
# # history_1, test_acc_1, test_loss_1 = train_and_evaluate_model(model_1, "Model Exp 1", epochs=25)
# # save_model_all_formats(model_1, "model_exp_1")
# # results["Model Exp 1"] = {"accuracy": test_acc_1, "loss": test_loss_1}

# # # Model Experiment 2
# # model_2 = create_model_exp2()
# # history_2, test_acc_2, test_loss_2 = train_and_evaluate_model(model_2, "Model Exp 2", epochs=25)
# # save_model_all_formats(model_2, "model_exp_2")
# # results["Model Exp 2"] = {"accuracy": test_acc_2, "loss": test_loss_2}

# # # Model Experiment 3
# # model_3 = create_model_exp3()
# # history_3, test_acc_3, test_loss_3 = train_and_evaluate_model(model_3, "Model Exp 3", epochs=25)
# # save_model_all_formats(model_3, "model_exp_3")
# # results["Model Exp 3"] = {"accuracy": test_acc_3, "loss": test_loss_3}

# # # Model Experiment 4
# # model_4 = create_model_exp4()
# # history_4, test_acc_4, test_loss_4 = train_and_evaluate_model(model_4, "Model Exp 4", epochs=25)
# # save_model_all_formats(model_4, "model_exp_4")
# # results["Model Exp 4"] = {"accuracy": test_acc_4, "loss": test_loss_4}

# # # Model Experiment 5
# # model_5 = create_model_exp5()
# # history_5, test_acc_5, test_loss_5 = train_and_evaluate_model(model_5, "Model Exp 5", epochs=25)
# # save_model_all_formats(model_5, "model_exp_5")
# # results["Model Exp 5"] = {"accuracy": test_acc_5, "loss": test_loss_5}

# # # Model Experiment 6
# # model_6 = create_model_exp6()
# # history_6, test_acc_6, test_loss_6 = train_and_evaluate_model(model_6, "Model Exp 6", epochs=25)
# # save_model_all_formats(model_6, "model_exp_6")
# # results["Model Exp 6"] = {"accuracy": test_acc_6, "loss": test_loss_6}

# # # Model Experiment 7
# # model_7 = create_model_exp7()
# # history_7, test_acc_7, test_loss_7 = train_and_evaluate_model(model_7, "Model Exp 7", epochs=25)
# # save_model_all_formats(model_7, "model_exp_7")
# # results["Model Exp 7"] = {"accuracy": test_acc_7, "loss": test_loss_7}

# # # Model Experiment 8
# # model_8 = create_model_exp8()
# # history_8, test_acc_8, test_loss_8 = train_and_evaluate_model(model_8, "Model Exp 8", epochs=25)
# # save_model_all_formats(model_8, "model_exp_8")
# # results["Model Exp 8"] = {"accuracy": test_acc_8, "loss": test_loss_8}

# # ================================================================
# # 7. RESULTS COMPARISON
# # ================================================================

# def compare_all_results():
#     """
#     Compare results from all models
#     """
#     if not results:
#         print("No results to compare. Please train at least one model first.")
#         return
    
#     print("\n" + "="*80)
#     print("MODEL COMPARISON RESULTS")
#     print("="*80)
    
#     # Create comparison DataFrame
#     comparison_df = pd.DataFrame(results).T
#     comparison_df = comparison_df.round(4)
    
#     print(comparison_df)
    
#     # Plot comparison
#     plt.figure(figsize=(15, 6))
    
#     # Accuracy comparison
#     plt.subplot(1, 2, 1)
#     models = list(results.keys())
#     accuracies = [results[model]['accuracy'] for model in models]
    
#     bars = plt.bar(range(len(models)), accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange', 'cyan', 'pink'][:len(models)])
#     plt.xlabel('Models')
#     plt.ylabel('Test Accuracy')
#     plt.title('Model Comparison - Test Accuracy')
#     plt.xticks(range(len(models)), [f'Exp {i+1}' for i in range(len(models))], rotation=45)
#     plt.ylim(0, 1)
    
#     # Add accuracy labels on bars
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
#                 f'{height:.3f}', ha='center', va='bottom')
    
#     # Loss comparison
#     plt.subplot(1, 2, 2)
#     losses = [results[model]['loss'] for model in models]
    
#     bars = plt.bar(range(len(models)), losses, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange', 'cyan', 'pink'][:len(models)])
#     plt.xlabel('Models')
#     plt.ylabel('Test Loss')
#     plt.title('Model Comparison - Test Loss')
#     plt.xticks(range(len(models)), [f'Exp {i+1}' for i in range(len(models))], rotation=45)
    
#     # Add loss labels on bars
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
#                 f'{height:.3f}', ha='center', va='bottom')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Find best model
#     best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
#     print(f"\n🏆 Best Model: {best_model}")
#     print(f"   Test Accuracy: {results[best_model]['accuracy']:.4f}")
#     print(f"   Test Loss: {results[best_model]['loss']:.4f}")

# # ================================================================
# # 8. QUICK START EXAMPLE
# # ================================================================

# print("="*80)
# print("CNN CAT VS DOG CLASSIFICATION TEMPLATE")
# print("="*80)
# print("\nQuick Start Instructions:")
# print("1. Ensure your dataset is in '/content/dataset/combined' with 'cats' and 'dogs' folders")
# print("2. Uncomment the model experiment you want to run")
# print("3. Run the cells to train and evaluate the model")
# print("4. Models will be saved in SavedModel, TF-Lite, and TFJS formats")
# print("\nAll models use:")
# print("- Sequential architecture with Conv2D and MaxPooling2D layers")
# print("- Target accuracy >85% on both training and testing")
# print("- Complete visualization of training progress")
# print("- Comprehensive evaluation metrics")

# # Example: Train Model Experiment 1
# print("\n" + "="*50)
# print("EXAMPLE: Training Model Experiment 1")
# print("="*50)

# # Uncomment below to run example
# model_1 = create_model_exp1()
# history_1, test_acc_1, test_loss_1 = train_and_evaluate_model(model_1, "Model Exp 1 - CNN Architecture (Using 32 Neurons in Conv Layer)", epochs=25)

# # Save model in all formats
# save_model_all_formats(model_1, "model_exp_1")

# # Store results
# results["Model Exp 1"] = {"accuracy": test_acc_1, "loss": test_loss_1}

# # Compare results (if you have trained multiple models)
# # compare_all_results()

# print("\n✅ Template setup complete! You can now train other models by uncommenting the respective sections.")