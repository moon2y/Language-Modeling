# Language-Modeling
Character-Level Language Models(Karphthy's article)
인공신경망과 딥러닝 과제
  1. vanilla RNN
  2. LSTM

## 파일 설명
- `main.py`: 모델 학습 & 테스트 + 샘플 생성
- `dataset.py`: shakespeare_train.txt 데이터셋 전처리
- `model.py`: vanilla RNN & LSTM 모델 구현
- generate.py : 학습된 모델로 샘플 생성

## 모델 설명
### vanilla RNN
![vanilla_rnn](https://github.com/moon2y/Language-Modeling/assets/88147264/25119bf3-7b9b-448c-bd08-0bff064a525b)

### LSTM
![lstm](https://github.com/moon2y/Language-Modeling/assets/88147264/1a569403-7ef5-4f4d-9922-adc1f1a3939e)

## 학습결과
### Hyper Parameters
- batch_size = 64
- hidden_size = 256
- n_layers -> 2 or 4
- learning_rate = 0.001
- n_epochs = 10
  
### n_layer = 2
![nl2](https://github.com/moon2y/Language-Modeling/assets/88147264/e4b62065-867f-4e32-9476-2fa83a1b3b99)


### n_layer = 4
![nl4](https://github.com/moon2y/Language-Modeling/assets/88147264/59cf8452-8f47-4d4d-af10-17f94b23a782)

### Train Loss
|                    | n_layer = 2 | n_layer = 4  |
|--------------------|-------------|--------------|
| vanilla RNN        | 0.9663      | 0.8519       |
| LSTM               | 0.5309      | 0.4713       |

### Test Loss
|                    | n_layer = 2 | n_layer = 4  |
|--------------------|-------------|--------------|
| vanilla RNN        | 0.9878      | 0.8759       |
| LSTM               | 0.5632      | 0.5028       |


## LSTM 생성 결과
### Hyper Parameters
- seed_characters_list = ["ROMEO:", "JULIET:", "HAMLET:", "OTHELLO:", "MACBETH:"]
- temperatures = [0.5, 0.7, 1.0, 1.2, 1.5]
- gen_length = 100

### n_layer = 2
- seed_characters_list = "ROMEO:"
  - (Temperature 0.5):
    ROMEO:
    If it be honour with you. Therefore, be gone.
    GLOUCESTER:
    To thee, that hast nor honesty nor grace
  - (Temperature 0.7):
    ROMEO:
    Yet one aid thank you, so did very
    many of that: you must die, my lord.
    CLARENCE:
    O, do not slande
  - (Temperature 1.0):
    ROMEO:
    The word, I am after!
    AUFIDIUS:
    I was moved withal.
    CORIOLANUS:
    No, sir,'twas not a state
    To one 
  - (Temperature 1.2):
    ROMEO:
    Ay, it, that it see my power and the tables was thee. Clarence, and human prize
    But his orthy tribu
  - (Temperature 1.5):
    ROMEO:
    If it were in France.
    YUCKINGHAM:
    Northumberland, that do I told the multited away with me?
    HASTI

- seed_characters_list = "JULIET:"
  - (Temperature 0.5):
    JULIET:
    My Lady Grey his wife, Clarence, 'tis she
    That tempers hims and legs, by this state,
    That could be 
  - (Temperature 0.7):
    JULIET:
    My gracious sovereign.
    KING RICHARD III:
    Art thou, indeed, madam?
    VALERIA:
    In troth, I think she 
  - (Temperature 1.0):
    JULIET:
    Oh, what should you tongues.
    HASTINGS:
    I know they do, sir, hear me speak.
    CORIOLANUS:
    You bland 
  - (Temperature 1.2):
    JULIET:
    Repary lives.
    QUEEN ELIZABETH:
    Hais the thus are too absolute;
    Though there disions; and with thy 
  - (Temperature 1.5):
    JULIET:
    What, my Lord of Derby?
    DERY:
    You must not deal upon us
    His in the wars wifely:
    Then shall we he's

- seed_characters_list = "HAMLET:"
  - (Temperature 0.5):
    HAMLET:
    Kind my lovely I think it is our way,
    I must be held; 'tis body bears, which shall but with here,
    T
  - (Temperature 0.7):
    HAMLET:
    Fir, you shall perceive
    Whether I blush or no: how; 'tis a soldier,
    Rather than envy you.
    COMINIUS
  - (Temperature 1.0):
    HAMLET:
    But had a tain thou in free heart's love to greet the tender princes.
    Daughter, well met.
    LADY ANN
  - (Temperature 1.2):
    HAMLET:
    Five, until without him; how the price of one fair work, I fear me.--Pray, your son, of me,
    But ona
  - (Temperature 1.5):
    HAMLET:
    O, then, I see us haunt, weakness! I will not take
    His popular of your last exercise;
    Cannot like t

- seed_characters_list = "OTHELLO:"
  - (Temperature 0.5):
    OTHELLO:
    MENENIUS:
    Why, my pretty York? I pray thee, let me have
    Some patience, never known before
    But to m
  - (Temperature 0.7):
    OTHELLO:
    MENENIUS:
    I'll undertake 't:
    I think he was young,
    Shouting their emulation.
    MENENIUS:
    Lo, citize
  - (Temperature 1.0):
    OTHELLO:
    His bloody brow! O Jupiter, no bloody minister,
    When gallant-springing brave Plantagenet
    Led in the
  - (Temperature 1.2):
    OTHELLO:
    Roman:
    The wide closk o' call'd forth free, then remember
    A foest truth, for drawn rest.
    DUCHESS 
  - (Temperature 1.5):
    OTHELLO:
    I'll inrquies all posterity,
    Or let us loset;
    And follow Marcius; I do know the Taquest you think s
    
- seed_characters_list = "MACBETH:"
  - (Temperature 0.5):
    MACBETH:
    You have made fair work, I fear me.--Pray, your news?--
    If Marcius should be join'd with Volscians,
  - (Temperature 0.7):
    MACBETH:
    Ay murder me for this, perform a part
    Thou hast not think it. Hark! what then?
    First Citizen:
    We h
  - (Temperature 1.0):
    MACBETH:
    Ay, my life, some tory whose power well wrath or no! You are some bark
    your grace?
    QUEEN ELIZABETH
  - (Temperature 1.2):
    MACBETH:
    Here, Hastings;
    When they account his head and
    though their heart to send you have deserved the pla
  - (Temperature 1.5):
    MACBETH:
    Pitcher court together;
    Madam, even, good Servingman: would all wind of sovereigntaly?
    MENENIUS:
    I
