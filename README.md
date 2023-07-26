# COMP 7705: Trading Bots with Reinforcement Learning  
## Basic Information  

This repository is the implementation of RL model of our COPM 7705 Final Project, namely **Trading Bots with Reinforcement Learning**.  
This project is supervised by **Dr. H.F. Ting**  
Our project group members include:  
- ZHANG Weiwen (3036034734, wwzhang@connect.hku.hk)  
- WANG Haonan (3036034576)  
- XIA Yunlei (3036035659)  
- XIN Hong (3036031914)  
From Department of Computer Science, Faculty of Engineering, The University of Hong Kong.  

Please note that this repository only contains the training (and evaluation) section of our algorithm. Other sections of our project, includes backend systems and frontend interfaces are not included in this repository.  

Also please note that our training code is constructed based on the skeleton from [source code reference](https://github.com/pskrunner14/trading-bot). Stock datasets have already been included, and the [bitcoin dataset](https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd) is to be placed at ./dataset/archive/  


## Training Instruction   
To train and validate the model, the script is shown as below:  
1. DQN training for Google Dataset:
``` 
python train.py ./dataset/data/GOOG.csv ./dataset/data/GOOG_2019.csv \
--strategy dqn --episode-count 50 --model-name dqn_ggl \
--sample-rate 0.3 --balance 100000.0
```

2. DQN training for Apple Dataset:
``` 
python train.py ./dataset/data/AAPL.csv ./dataset/data/AAPL_2018.csv \
--strategy dqn --episode-count 50 --model-name dqn_apl \
--sample-rate 0.3 --balance 100000.0
```

3. DQN training for BTC Dataset:
```
python train.py ./dataset/archive/BTC-2021min.csv ./dataset/archive/BTC-2021min.csv \
--strategy dqn --episode-count 50 --model-name dqn_btc \
--sample-rate 0.005 --balance 100000.0
```

4. T-DQN training for Google Dataset:
``` 
python train.py ./dataset/data/GOOG.csv ./dataset/data/GOOG_2019.csv \
--strategy t-dqn --episode-count 50 --model-name tdqn_ggl \
--sample-rate 0.3 --balance 100000.0

```

5. T-DQN training for Apple Dataset:
``` 
python train.py ./dataset/data/AAPL.csv ./dataset/data/AAPL_2018.csv \
--strategy t-dqn --episode-count 50 --model-name tdqn_apl \
--sample-rate 0.3 --balance 100000.0
```

6. T-DQN training for BTC Dataset:
```
python train.py ./dataset/archive/BTC-2021min.csv ./dataset/archive/BTC-2021min.csv \
--strategy t-dqn --episode-count 50 --model-name tdqn_btc \
--sample-rate 0.005 --balance 100000.0
```

7. Double-DQN training for Google Dataset:
``` 
python train.py ./dataset/data/GOOG.csv ./dataset/data/GOOG_2019.csv \
--strategy double-dqn --episode-count 50 --model-name doubledqn_ggl \
--sample-rate 0.3 --balance 100000.0
```

8. Double-DQN training for Apple Dataset:
``` 
python train.py ./dataset/data/AAPL.csv ./dataset/data/AAPL_2018.csv \
--strategy double-dqn --episode-count 50 --model-name doubledqn_apl \
--sample-rate 0.3 --balance 100000.0
```

9. Double-DQN training for BTC Dataset:
```
python train.py ./dataset/archive/BTC-2021min.csv ./dataset/archive/BTC-2021min.csv \
--strategy double-dqn --episode-count 50 --model-name doubledqn_btc \
--sample-rate 0.005 --balance 100000.0
```
  
Some of our experimental logs and visualization results are placed in the ```run.ipynb```.  
