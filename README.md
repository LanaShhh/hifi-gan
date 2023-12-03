# hifi-gan

Данная модель представляет из себя реализацию HiFi-GAN, описанную в статье HiFi-GAN: Generative Adversarial Networks for
Efficient and High Fidelity Speech Synthesis (https://arxiv.org/pdf/2010.05646.pdf). 
Модель реализована в рамках курса НИУ ВШЭ Deep Learning for Audio 2023.

## Инструкция по использованию кода

- Скачиваем данный git-репозиторий

~~~
git clone https://github.com/LanaShhh/hifi-gan.git
mv hifi-gan/* .
~~~

- Скачиваем все необходимые библиотеки

~~~
pip install -r requirements.txt
~~~

### Обучение 

~~~
python3 training.py wandb_login_key
~~~

wandb_login_key - обязательный аргумент, ключ для авторизации в wandb (https://wandb.ai), можно получить по ссылке https://wandb.ai/authorize

Результат - run в wandb, папка checkpoints с последним чекпоинтом модели, папка train_audio со сгенерированным тестовым аудио на последнем чекпоинте.

### Инференс 

~~~
python3 inference.py model_state_dict_path
~~~

model_state_path - обязательный параметр, путь до модели. 

Итоговый чекпоинт модели автора - TBD ссылка.

Для тестирования чекпоинта выше необходимо скачать файл и положить его в папку checkpoints: "./checkpoints.checkpoint.pth.tar"

~~~
python3 inference.py ./checkpoints/checkpoint.pth.tar
~~~

Результат - папка results с генерациями трех аудио на основе mel-спектрограмм аудио из папки ./test_audios

## Отчет 

Отчет в wandb, с описанием работы, графиками функций ошибок, выводами и сгенерированными аудио во время обучения - TBD ссылка

Ссылка на итоговый run в wandb - TBD ссылка


## Итоговая генерация

Для оценки качества полученной модели производилась генерация аудио из папки ./test_audios.

Результаты генерации находятся в папке ./results.

## Источники

- Jungil Kong, Jaehyeon Kim, Jaekyoung Bae. HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech. https://arxiv.org/pdf/2010.05646.pdf

- Курс Deep Learning for Audio, НИУ ВШЭ, ПМИ, 2023. https://github.com/XuMuK1/dla2023

- PyTorch documentation. https://pytorch.org/docs/stable/index.html




