import pandas as pd
import logging
from sklearn.utils import Bunch

from pathlib import Path
import tqdm
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image

logger = logging.getLogger(__name__)

def get_files(path):
    """
    Возвращает список файлов по указанному пути.

    :param path: Путь к файлу или директории.
    :type path: str or pathlib.Path
    :return: Список путей:
             - Если передан файл — возвращает список из одного элемента.
             - Если передана директория — возвращает все файлы в ней (рекурсивно).
             - В остальных случаях — пустой список.
    :rtype: list[pathlib.Path]

    .. note::
        Метод использует :meth:`pathlib.Path.rglob` для рекурсивного поиска файлов.

    Пример:

    .. code-block:: python

        files = get_files("data/images/")
        print([f.name for f in files])
    """
    p = Path(path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        return list(p.rglob("*"))
    else:
        return []


class DataFactory:
    @staticmethod
    def load_dataset(dataset_config):
        """
        Загружает датасет в соответствии с конфигурацией.

        :param dataset_config: Конфигурация датасета, должна содержать ключ ``'type'`` с допустимыми значениями:
                               - ``'csv'``: загрузка из CSV-файла
                               - ``'images'``: загрузка изображений из директории
        :type dataset_config: dict
        :return: Объект :class:`sklearn.utils.Bunch` с атрибутами:
                 - ``data``: признаки (массив данных)
                 - ``target``: метки (если доступны, иначе ``None``)
        :rtype: sklearn.utils.Bunch
        :raises ValueError: Если указан неподдерживаемый тип датасета или произошла ошибка загрузки/валидации.

        .. note::
            Выбор загрузчика осуществляется по значению ``dataset_config['type']``.
            Поддерживаются только типы ``'csv'`` и ``'images'``.

        .. seealso::
            :meth:`_load_csv`, :meth:`_load_images` — внутренние методы загрузки конкретных типов данных.
        """
        ds_type = dataset_config.get("type")

        if ds_type == "csv":
            return DataFactory._load_csv(dataset_config)

        elif ds_type == "images":
            return DataFactory._load_images(dataset_config)

        else:
            raise ValueError(f"Unsupported dataset type: {ds_type}")

    @staticmethod
    def _load_csv(dataset_config):
        """
        Загружает табличные данные из CSV-файла.

        :param dataset_config: Конфигурация, содержащая:
                               - ``path`` (str): путь к CSV-файлу
                               - ``target`` (str): имя целевой колонки
        :type dataset_config: dict
        :return: Объект :class:`sklearn.utils.Bunch` со следующими атрибутами:
                 - ``data``: признаки в виде 2D numpy-массива
                 - ``target``: метки в виде 1D numpy-массива
                 - ``feature_names``: список имён признаков (столбцов без целевой переменной)
        :rtype: sklearn.utils.Bunch
        :raises ValueError:
            - Если не указан ``path`` или ``target``.
            - Если файл не может быть прочитан.
            - Если целевая колонка отсутствует в файле.

        .. note::
            Использует :func:`pandas.read_csv` для загрузки данных.

        Пример конфигурации:

        .. code-block:: python

            config = {
                "type": "csv",
                "path": "data/train.csv",
                "target": "label"
            }
            dataset = DataFactory._load_csv(config)
        """
        ds_type = dataset_config.get("type")
        if ds_type != "csv":
            raise ValueError("DataFactory currently supports only 'csv' type datasets.")

        path = dataset_config.get("path")
        if not path:
            raise ValueError("For CSV datasets, the configuration must include a 'path'.")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file at '{path}': {e}")

        target = dataset_config.get("target")
        if not target:
            raise ValueError("The configuration must specify the 'target' column name.")

        if target not in df.columns:
            raise ValueError(f"The target column '{target}' was not found in the CSV file.")

        X = df.drop(columns=[target]).values
        y = df[target].values

        feature_names = list(df.drop(columns=[target]).columns)

        logger.info(f"Dataset loaded successfully from {path} with {len(df)} samples.")
        return Bunch(data=X, target=y, feature_names=feature_names)

    @staticmethod
    def _load_images(config):
        """
        Загружает изображения из указанной директории.

        :param config: Конфигурация, содержащая:
                       - ``path`` (str): путь к директории с изображениями
        :type config: dict
        :return: Объект :class:`sklearn.utils.Bunch` с атрибутами:
                 - ``data``: numpy-массив изображений формы ``(N, C, H, W)``, нормализованный в диапазон [0, 1]
                 - ``target``: всегда ``None`` (метки не загружаются)
        :rtype: sklearn.utils.Bunch
        :raises ValueError: Если указанный путь не существует.
        :raises FileNotFoundError: Если директория не найдена.

        .. note::
            Поддерживаемые форматы: ``.jpg``, ``.jpeg``, ``.png``, ``.gif``, ``.bmp``, ``.tiff``.

        .. warning::
            Ошибки при загрузке отдельных изображений логируются, но не прерывают выполнение.
            Некорректные файлы просто пропускаются.

        .. note::
            Все изображения:
            - Преобразуются в RGB.
            - Изменяются до размера 640x640 с помощью бикубической интерполяции.
            - Центрируются.
            - Преобразуются в тензоры и затем в numpy-массивы с типом ``float32``.

        Процесс отображается с прогресс-баром с помощью ``tqdm``.
        """
        transform = transforms.Compose([
            transforms.Resize(640, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(640),
            transforms.ToTensor()
        ])

        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
        img_dir = Path(config.get("path"))

        if not img_dir.exists():
            raise ValueError("Provided image path does not exist.")

        images = []
        with tqdm.tqdm(get_files(img_dir), desc="Loading images") as pbar:
            for img_path in pbar:
                if img_path.suffix.lower() in image_extensions:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = transform(img).numpy()
                        images.append(img)
                    except Exception as e:
                        logger.error(e)

        logger.info(f"Total images loaded: {len(images)}")
        images = np.array(images).astype(np.float32)

        return Bunch(
            data=images,
            target=None,
        )