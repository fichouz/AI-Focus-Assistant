import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


DATA_DIR = "data"


def read_bin_file(path):
    data = np.fromfile(path, dtype=np.int16)

    usable_len = (len(data) // 6) * 6
    data = data[:usable_len]

    imu = data.reshape(-1, 6)

    return imu


def extract_features(imu):
    imu = imu.astype(np.float64)

    acc = imu[:, 0:3].astype(np.float64)
    gyro = imu[:, 3:6].astype(np.float64)

    features = {}

    names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    for i, name in enumerate(names):
        values = imu[:, i]

        features[f"{name}_mean"] = np.mean(values)
        features[f"{name}_std"] = np.std(values)
        features[f"{name}_min"] = np.min(values)
        features[f"{name}_max"] = np.max(values)

    acc_magnitude = np.sqrt(acc[:, 0]**2 + acc[:, 1]**2 + acc[:, 2]**2)
    features["acc_magnitude_mean"] = np.mean(acc_magnitude)
    features["acc_magnitude_std"] = np.std(acc_magnitude)

    gyro_magnitude = np.sqrt(gyro[:, 0]**2 + gyro[:, 1]**2 + gyro[:, 2]**2)
    features["gyro_magnitude_mean"] = np.mean(gyro_magnitude)
    features["gyro_magnitude_std"] = np.std(gyro_magnitude)

    features["stability"] = np.std(acc_magnitude) + np.std(gyro_magnitude)

    return features


def create_dataset():
    rows = []

    for label in ["focus", "sleepy", "distracted"]:
        folder = os.path.join(DATA_DIR, label)

        if not os.path.exists(folder):
            print(f"Mapa ne obstaja: {folder}")
            continue

        for filename in os.listdir(folder):
            if filename.endswith(".bin"):
                path = os.path.join(folder, filename)

                imu = read_bin_file(path)

                if imu.shape[0] == 0:
                    print(f"Datoteka je prazna ali neveljavna: {filename}")
                    continue

                features = extract_features(imu)

                features["label"] = label
                features["file"] = filename

                rows.append(features)

    df = pd.DataFrame(rows)

    return df


def plot_class_distribution(df):
    counts = df["label"].value_counts()

    plt.figure()
    counts.plot(kind="bar")
    plt.xlabel("Razred")
    plt.ylabel("Stevilo datotek")
    plt.title("Stevilo vzorcev po razredih")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("graf_razredi.png")
    plt.show()


def plot_confusion_matrix(cm, labels):
    plt.figure()
    plt.imshow(cm)
    plt.title("Matrika zamenjav")
    plt.xlabel("Napovedan razred")
    plt.ylabel("Pravi razred")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig("graf_matrika_zamenjav.png")
    plt.show()


def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_

    sorted_indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = importance[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_features, sorted_importance)
    plt.xlabel("Znacilke")
    plt.ylabel("Pomembnost")
    plt.title("Pomembnost znacilk pri Random Forest modelu")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("graf_pomembnost_znacilk.png")
    plt.show()


def main():
    print("Zacenjam pripravo podatkov...")

    df = create_dataset()

    if df.empty:
        print("Napaka: dataset je prazen. Preveri mapo data in BIN datoteke.")
        return

    print()
    print("Prvih nekaj vrstic dataseta:")
    print(df.head())

    print()
    print("Stevilo vzorcev po razredih:")
    print(df["label"].value_counts())

    df.to_csv("imu_features_dataset.csv", index=False)

    print()
    print("Dataset je shranjen kot imu_features_dataset.csv")

    plot_class_distribution(df)

    X = df.drop(columns=["label", "file"])
    y = df["label"]

    print()
    print("Delim podatke na ucno in testno mnozico...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("Ucim Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Model je naucen.")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print()
    print("Tocnost modela:")
    print(accuracy)

    print()
    print("Matrika zamenjav:")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print(cm)

    print()
    print("Porocilo klasifikacije:")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(cm, model.classes_)
    plot_feature_importance(model, X.columns)

    joblib.dump(model, "imu_random_forest_model.pkl")

    print()
    print("Model je shranjen kot imu_random_forest_model.pkl")
    print("Graf razredov je shranjen kot graf_razredi.png")
    print("Graf matrike zamenjav je shranjen kot graf_matrika_zamenjav.png")
    print("Graf pomembnosti znacilk je shranjen kot graf_pomembnost_znacilk.png")
    print()
    print("Naloga je koncana.")


if __name__ == "__main__":
    main()