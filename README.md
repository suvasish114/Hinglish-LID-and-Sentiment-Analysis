# Language Identification and Sentiment Analysis from Hindi English Code-Mixed Texts

This project is under active development for partial fullfillment of grade requirement for course code CS613-NLP at Indian Institute of Technology Gandhinagar.

### Download Model Checkpoints

Run the below command to download the model checkpoints using CLI. Some of the model require authentication. Make sure you have authenticated your device before accessing the model.

```sh
python3 lib/download_model_checkpoints.py
```

### Setup

This project is depended on various python pip packages. Its recommended to use a virtual enviornment to keep your current workspace clearn. Use the below command sequentially

```sh
python3 -m venv venv # skip if you already have virtual enviornment
source venv/bin/activate
pip install requirements.txt # one time
python3 download_model_checkpoints.py load_data.py # one time
python3 main.py # reproduction
```

> This project has been tested on CUDA12.4

### Authors

- Nishant Sharma - [rockbnishant](https://github.com/Rockbnishant)
- Durgesh Mishra - [durg3sh10](https://github.com/durg3sh10)
- Rahul - [rahulkumawat835](https://github.com/rahulkumawat835)
- Lavish Jangid - [lavish-j](https://github.com/lavish-j)
- Harshiddhi Pathak - [horikita-99](https://github.com/horikita-99)
- Anuj Tiwari - [anujjtiwari](https://github.com/anujjtiwari) 
- Harsh Krishnadev Dubey []()
- Suvasish Das - [suvasish114](https://github.com/suvasish114)

<!-- <br><br>

<a href="https://github.com/suvasish114/Hinglish-LID-and-Sentiment-Analysis/graphs/contributors"><img src="https://contrib.rocks/image?repo=suvasish114/Hinglish-LID-and-Sentiment-Analysis" /></a> -->
