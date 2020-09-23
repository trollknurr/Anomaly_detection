# Anomaly_detection

Clover R&D


Docker: https://hub.docker.com/repository/docker/markhubbatulin/anomaly_detection

Для тестирование используйте app_test.py. Ответ приходит в форме JSON

{

    'filename': ,
    
    'score': ,
    
    'status':
}

file - название пришедшего файла

score - уровень аномальности

status - True: аномальный, False: нормальный
