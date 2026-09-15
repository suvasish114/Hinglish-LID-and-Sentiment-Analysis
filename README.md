# Language Identification and Sentiment Analysis from Hindi English Code-Mixed Texts

This project is under active development for partial fullfillment of grade requirement for course code CS613-NLP at Indian Institute of Technology Gandhinagar.

### Enviornment setup

This project require specified python libraries. Make sure to use python enviornment for the experiments.

```sh
python3 -m venv venv # create environment (one time)
source venv/bin/activate # activate your enviornment (linux)
pip install -r requirements.txt # install all required packages (one time)
```

If you install any new packages, make sure to add those in `requirements.txt` by using

```sh
pip freeze > requirements.txt
```
> This project has been tested on CUDA12.4

### Download dataset

For all the experiment the datasets, models and hyperparameters remain same. Please use the data generation script to download and process the SentiMix dataset. Alongside, use `eval.py` to generate the result. Use below command to download and process the dataset.

```sh
python3 get_sentimix.py # one time 
```

This will download all the required files from the remote server and process it to `csv` format and store them in your current folder's `dataset/SentiMix/`. For evaluation use 

```sh
python3 eval.py
```

For experiment specific details, use branches `exp0`, `exp1`, ...

### Authors

- Parth Dangi - [parthgdangi](https://github.com/parthgdangi)
- Nishant Sharma - [rockbnishant](https://github.com/Rockbnishant)
- Durgesh Mishra - [durg3sh10](https://github.com/durg3sh10)
- Rahul Kumawat - [rahulkumawat835](https://github.com/rahulkumawat835)
- Lavish Jangid - [lavish-j](https://github.com/lavish-j)
- Harshiddhi Pathak - [horikita-99](https://github.com/horikita-99)
- Anuj Tiwari - [anujjtiwari](https://github.com/anujjtiwari) 
- Harsh Krishnadev Dubey [Hrshhh](https://github.com/Hrshhh)
- Suvasish Das - [suvasish114](https://github.com/suvasish114)


### Contributors

<a href="https://github.com/suvasish114/Hinglish-LID-and-Sentiment-Analysis/graphs/contributors"><img src="https://contrib.rocks/image?repo=suvasish114/Hinglish-LID-and-Sentiment-Analysis"/></a> 
