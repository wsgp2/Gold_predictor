"""
Модуль для загрузки конфигурации и переменных окружения
"""
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_environment_variables():
    """
    Загружает переменные окружения из файла .env
    
    Returns:
        bool: True если переменные успешно загружены, иначе False
    """
    # Ищем .env файл в корневой директории проекта
    base_dir = Path(__file__).parent.parent.absolute()
    env_path = base_dir / '.env'
    
    if env_path.exists():
        # Загружаем переменные окружения из .env файла
        load_dotenv(dotenv_path=str(env_path))
        logger.info(f"Переменные окружения загружены из {env_path}")
        return True
    else:
        logger.warning(f"Файл .env не найден в {base_dir}. Используем системные переменные окружения.")
        return False

def get_config(config_path):
    """
    Загружает конфигурацию из JSON файла и обогащает ее переменными окружения
    
    Args:
        config_path (str): Путь к файлу конфигурации
    
    Returns:
        dict: Словарь с конфигурацией
    """
    # Проверяем наличие файла конфигурации
    if not os.path.exists(config_path):
        logger.error(f"Файл конфигурации не найден: {config_path}")
        return {}
    
    # Загружаем конфигурацию из JSON файла
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Конфигурация загружена из {config_path}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        return {}
    
    # Загружаем переменные окружения
    load_environment_variables()
    
    # Обновляем конфигурацию переменными окружения
    if os.getenv('TELEGRAM_TOKEN'):
        config['telegram_token'] = os.getenv('TELEGRAM_TOKEN')
    
    if os.getenv('TELEGRAM_CHAT_ID'):
        config['telegram_chat_id'] = os.getenv('TELEGRAM_CHAT_ID')
    
    # Добавляем переменные для Bybit
    config['bybit_api_key'] = os.getenv('BYBIT_API_KEY', '')
    config['bybit_api_secret'] = os.getenv('BYBIT_API_SECRET', '')
    
    return config

def save_config(config, config_path):
    """
    Сохраняет конфигурацию в JSON файл, исключая секретные данные
    
    Args:
        config (dict): Словарь с конфигурацией
        config_path (str): Путь для сохранения файла конфигурации
    
    Returns:
        bool: True если конфигурация успешно сохранена, иначе False
    """
    # Создаем копию конфигурации без секретных данных
    safe_config = config.copy()
    
    # Удаляем секретные данные
    if 'bybit_api_key' in safe_config:
        del safe_config['bybit_api_key']
    if 'bybit_api_secret' in safe_config:
        del safe_config['bybit_api_secret']
    
    # Сохраняем конфигурацию в JSON файл
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(safe_config, f, indent=4, ensure_ascii=False)
        logger.info(f"Конфигурация сохранена в {config_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении конфигурации: {e}")
        return False

# При импорте модуля загружаем переменные окружения
_ = load_environment_variables()
