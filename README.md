# BERT 코드 분석
## 한국어 BERT로 간단하게 해보기
## 데이터 생성
### 개요
*   BERT에 집어넣을 데이터를 생성하는 코드 (create_pretraining_data.py)
*   입력: sample_text.txt◾
*   엔터로 문단을 구분하고 문단별로 다른 것을 말하는 내용이 들어있는 것으로 생각하면 됨.
*   예를 들어, 첫 번째 문단은 야구기사, 두 번쨰 문단은 정치기사 등등
*   이것으로 BERT의 NSP와 같은 pre-training을 하기 위한 data을 만듬
*   출력: 텐서플로우에서 tfrecord 라고 하는 파일로 생성이 됨
*   TFRecord 파일은 텐서플로우의 학습 데이타 등을 저장하기 위한 바이너리 데이타 포맷으로, 구글의 Protocol Buffer 포맷으로 데이타를 파일에 Serialize 하여 저장한다.
*   보통 데이터가 너무 많은 경우, 매번 불러 읽어들여서 하는 것이 아니라 tfrecord로 처리를 해둬서 저장을 하고 다시 불러 읽어들인다.
*   BERT는 대용량 데이터를 사용하기 때문 tfrecord을 사용해야 한다고 보면 됨


### 내용
*   다양한 hyperparametr을 받음◾input_file: 위에서 설명
*   output_file: 위에서 설명
*   vocab_file: 사전에 구축되어 있는 voacb.txt◾
*   물론 영어권은 기본적인 사전이 다 구축되어 있음
*   한국어 BERT을 만든다고 생각하면 이것부터 구축해야함.
*   do_lower_case: 대소문자 구별한 것인지 (구별(False): uncased, 비구별(True): cased)
*   do_whole_word_mask: per-WordPiece masking 대신에 whole word masking을 쓸 것인가에 대한 것 (default: False)
*   max_seq_length: 최대 문장 길이 (넘어가면 자르게 됨)
*   max_predictions_per_seq: 문장 당 가장 많이 masked LM prediction의 숫자 (정확히 무엇을 의미?)
*   random_seed: 데이터 생성할 때 사용이 되는데 신경 안써도됨. 아무거나 넣으면 됨 (default: 12345)
*   dupe_factor: 입력 데이터에 mask을 씌울 때 몇 가지 방법으로 씌울 것인가에 대한 것으로 생각하면 됨 (default: 10)
*   masked_lm_prob: masked LM 확률 (default: 0.15)
*   short_seq_prob: Probability of creating sequences which are shorter than the maximum length. (정확히 무엇을 의미?) (default: 0.1)
*   코드 상에서 최종 출력물에 데이터를 instances 변수로 바꾼다음 저장해둔다고 보면 된다.◾
*   결국 한국어 vocab, tokenizer을 이용하여 Instances을 만들어내기만 하면 된다.◾Instances 구조를 살펴보자
*   즉 insatnces안에는 데이터로부터 만들어진 <CLS> A_tokens <SEP> B_tokens <SEP> 식의 tokens이 저장이 된다.
*   문장 중간 중간 <MASK>로 치환된 것을 볼 수가 있다.◾<MASK>토큰의 정답인 masked_lm_labels가 있다.
*   또한 그것에 대한 token 위치인 segment_ids가 있다.
*   pre-training 단계에서 <MASK> 예측이외의 학습인 NSP가 있는데 이를 위하여 is_random_next의 key가 있고 False, True의 값을 가진다.
*   우리가 한국어 BERT을 만들기 위해서 준비해야할 것은 한글 코퍼스 vocab.txt와 한글 tokenizer을 이용한 코드 수정이라고 보면 될 것이다.

### 실제 생성 과정
*   한글 샘플 텍스트를 만들기 위하여 네이버기사 몇 개를 긁어서 생성◾
*   Tokenizer은 Mecab 사용◾Mecab+python 사용하는 것은 https://sens.tistory.com/445 을 참고하여서 진행하면 됨.
*   Mecab을 이용하여 사전 구축◾사전은 형태소는 고려하지 않고 글자로만 구성하였음.
*   스페셜 토큰으로 BERT와 똑같이 <PAD>, <UNK>, <CLS>, <SEP>, <MASK> 을 초기에 추가함.
*   총 vocab 수는 265개
*   BERT에서 기본으로 사용하는 tokenization은 다음의 기능을 포함한다.◾tokenizer.tokenize: 토크나이저 하는 기능
*   tokenizer.convert_tokens_to_ids: 토큰의 index을 가져오는 기능
*   okenizer.vocab.keys: vocab_list을 뽑는 기능
*   Mecab은 토크나이저 기능을 빼고는 없기 때문에 이 부분을 만들어 코드를 수정하였음
*   코드를 수정하는 방법은 2가지로 할 수가 있음
*   1) create_data 코드에서 tokenizer에 관련된 함수를 직접 수정하는 방법
*   2) tokenization.py 자체에 mecab tokenizer을 추가하여 수정
*   1번 방법으로 먼저 바꿔서 돌렸지만 당연 2번 방법이 깔끔해서 2번으로 하는 게 낫다
*   왜냐하면 추후 tokenization을 이용하여 연결되는 코드들이 있을 수 있으므로 그때마다 코드를 매번 수정하기는 복잡하기 때문
*   출력은 tfrecord의 파일이 생성

## Pre-training
### 개요
*   1번 단계에서 tfrecord 파일을 만들었다. (from create_pretraining_data.py)
*   이제 BERT을 학습시키면되고 사용할코드는 run_pretraining.py 이다.
*   입력: data.tfrecord
*   출력: 학습된 BERT 모델

### 내용
*   몇 개의 인자를 받게 되어있는데 살펴보자◾bert_config_file: BERT 모델이 구성되는 여러 가지 파라미터를 적어둔 json 파일이다.◾ → 
*   위의 왼쪽 스샷은 BERT의 base 모델을 가져온 것이고 본인이 원하는 숫자로 바꿔서 학습해도 된다.
*   근데 한국어 BERT라고 굳이 다른 파라미터를 바꿀 필요는 없어보인다.
*   단 vocab_size을 바꿔줘야 되는데 샘플로 데이터로 만든 korea_vocab 사이즈 265로 바꾸면 된다.
*   input_file: 1번 단계에서 만들어운 tfrecord의 path를 적어주면 된다.
*   output_dir: 출력물이 정해질 위치
*   init_checkpoint: 맨 처음에는 당연히 None이다.◾만약 BERT 모델을 구글 colab 같은데서 학습한다면, 여기에 저장되는 위치를 적어주면 될 것이다.
*   max_seq_length: 입력 문장 길이가 최대 길이 넘어가면 자르게 되고 문장이 짧으면 padding 하는데 그 길이를 의미
*   max_predictions_per_seq: 문장당 최대 몇 번의 masked LM 예측할지를 의미
*   do_train: 학습할지 의미
*   do_eval: 중간에 eval 할지 의미
*   train_batch_size: 학습 배치 사이즈
*   eval_batch_size: eval 배치 사이즈
*   learning_rate: learning rate for Adam (default: 2e-5)
*   num_train_steps: 학습 steps
*   num_warmup_steps: warmup steps으로 optimization에 들어가는 변수이다. (default 20)
*   save_checkpoints_steps: 저장 체크포인트
*   iterations_per_loop: How many steps to make in each estimator call.
*   max_eval_steps: Maximum number of eval steps. (default 100)
*   use_tpu:  TPU 사용할지 말지◾이것을 True로 할거면 밑에 따라오는 몇 개의 TPU 설정 파라미터들을 수정해줘야한다.
*   그런데 TPU True해서 학습해보니 여기서도 error가 난다는... (BERT 실험의 finetuned 때 처럼..)

## Fine-tunning (MRC)

### 개요
*   이제 BERT 코드를 이용하여 fine-tunning을 해보자
*   squad으로 돌려보고 korquad로 돌려보면서 한국어 BERT로 MRC에 적용시켜보자.
*   한국어 BERT 자체를 pretraining으로 구축하기는 상당히 오래걸리므로, 오연택선임이 구축한 모델을 사용하였다.•구축한 모델은 mecab tokenizer을 사용하고 pos agging은 사용하지 않은 모델
*   vocab은 128000개이다.
*   단, 오선임이 수정한 run_squad 을 사용하지 않고 부분과 다르게 수정하였다.

### 내용
*   이 코드를 분석하면서 vocab_generation을 할 때, ##부분을 깜빡함을 인지하였다.◾단순히 run_pretraining을 할 때는 ##을 고려안해도 상관없다.
*   실제로 이것을 fine_tunning할 때도 코드를 커스터마이징해서 쓸 수도 있다.
*   하지만 기본적으로 BERT의 vocab 구성은 ##을 이용한다. (## 이용하는게 일반적)
*   즉 "사과나무" → "사과" + "##나무" 와 같게 쪼개지는 단어는 ##이 붙어버린다.
*   따라서 BERT 모델을 사용할 때 코드에 이런 것을 처리하는 코딩들이 들어있는데 한글어 버전에서 ##을 안쓰면, 이 부분을 수정해야한다.
*   또한 오선임이 구축한 모델에서도 ## 기법을 사용하였기 때문에 ##을 고려하여 tokenizer을 수정하였다.
*   모델을 학습하기 위해서 데이터 전처리를 하는 과정이 필요하다.◾즉 데이터로 미리 학습데이터를 tf_record로 만들어 둔다.
*   이것을 어떻게 만드냐에 따라서 방법이 여러가지가 있을 것이다.
*   위와 같이 tokenization.py을 수정하고 코드를 돌리면 된다.◾돌리다보면, 가끔식 error가 발생을 하는데 발생이유를 찾아보면 데이터 문제다.
*   찾은 error로는 text에 다음과 같이 띄어쓰기가 2번 있는 경우다.◾"이 텍스트는  샘플입니다."에서 "텍스트"와 "샘플"사이의 띄어기가 2번 있는 경우
*   혹은 양끝에 공백이 있는 경우다.◾"이 텍스트는 샘플입니다. "에서 "샘플입니다." 뒤에 공백이 있는 경우
*   이런 자잘한 error을 찾아줘서 수정해주면 된다.
*   시행착오◾맨처음에는 read_squad_examples 함수를 수정하는 방향으로 갔다.
*   왜냐하면, reqd_squad_examples을 영어버전으로 돌려보면, 띄어쓰기 단위로 token을 쪼개서 answer_start와 answer_end을 찾는다.
*   즉, 데이터에서 주어진 MRC에서 주어진 정답 answer은 character 단위의 위치이다.◾예)"정답은 6번째 입니다."에서 answer_position 6라는 것은 "번"의 위치를 말다. (띄어쓰기 포함)

*   이것을 띄어쓰기 단위로 token을 쪼개서 이것과 align을 시키는 것이다.
*   하지만 우리가 모델을 학습할 때는 사용한 tokenizer의 token 단위의 position을 찾아야 한다.◾따라서 meacb tokenizer에 맞게 수정코드를 짜면, 이 뒤에 사용되는 함수들을 수정해야한다.
*   하지만 뒷 부분을 수정하다보면  convert_examples_to_features 함수를 수정해줘야 한다.◾이 함수를 분석해보면 띄어쓰기로 쪼개진 token을 실제 사용하는 tokenizer의 subtoken으로 쪼개서 align을 맞춘다.
*   하지만 위에서 이미 mecab_tokenizer로 align을 시켰으니 이 부분을 적절히 삭제, 수정하면 된다.
*   하지만 굳이 이렇게 전체적으로 코드를 수정안해도 학습이 되니 시행착오로 남겨둔다..
*   단, 만약에 BERT모델을 huggingFace와 같은 라이브러리를 이용하여 사용한다면◾fine_tunning을 실제로 코드를 밑바닥부터 짜야하는데
*   이럴 때는 시행착오에서 겪었던 tokenizer에 맞는 answer position align을 직접 짜줘야한다.
*   또한 tensorflow-BERT는 fine-tunning을 할 때, BERT 뒷부분의 layer을 어떻게 짰는지 코드상으로는 확인이 어렵다.
*   이 부분은 estimator으로 감싸져 있는 것 같다.
*   따라서 pytorch-BERT을 사용할 떄는 논문을 참고해서 이 부분의 코드를 짜야한다.

### 결과
*   {"exact_match": 19.570488396259094, "f1": 80.1120485168468}◾f1은 잘나오는데 EM이 다른 korquad 모델에 비해 낮다
*   어떤 문제점일까? (한 번 확인해봐야겠음)◾이 모델을 5 epoch 정도 학습하면 50 이상의 EM이 나온다고 한다.
*   그리고 이러한 조사, 어미 등의 문제점은 두 가지 방법으로 해결할 수 있다.
*   1) 애초에 사용하는 토크나이징으로 조사하고 잘 분리되도록 하는 것 (POS tagging까지 사용한 tokenizer)◾그러면 조사 부분은 정답으로 학습이 안되니까 예측도 사단위로 끈어서 하게 됨.
*   2) 어떤 토크나이저를 사용하더라도 (e.x. BPE) 정답 부분에서 조사를 떼어주는 후처리를 적용
*   학습할 때, 2 epoch만 학습하고 pos tagging도 안사용하였기 때문일까?
*   성능을 위해 어미, 조사같이 의미를 담고있지 않는 끝부분을 후처리 하는 것을 넣으면 성능이 향상될 것 같다.
*   dev dataset의 예측 결과를 보면 다음과 같다.◾
*   첫 번째 단어가 answer이고 두 번째 단어가 prediction이다.
*   결과만 보면 학습은 제대로 된 것 같다. (즉 제대로된 학습 절차를 거친 것은 맞아보인다.)
*   샘플들을 보아도 F1은 높게 측정이 될 것이고 EM은 낮을 거 같은게 보인다.

## (변환) BERT Tensorflow 모델 Pytorch 모델로 사용하기 
### 개요
*   기본적으로 앞의 실험들은 (pretraining, fine-tunning) 텐서플로우로 진행하는 것이다.
*   구글에서 제공한 공식 텐서플로우 코드였고 버전은 TF 1.4로 1점대 버전이다.
*   이렇게 학습을 하기에는 다음과 같은 문제점이 있다고 판단한다.◾2점대 버전인 요즘 텐서플로우와는 사용방법의 거리가 있음
*   기존에 사용하던 fine-tunning에는 squad처럼 코드의 일부분을 수정하여 사용은 가능하나
*   가 원하는 뒷단의 layer 수정, 새로운 task에 fine-tunning 핸들링하는 것이 쉽지 않다.◾물론 구글식 텐서플로우 코드가 상당히 익숙하면 어렵지 않겠지만...
*   따라서 비교적 사용법이 쉬운 huggingface 방식으로 파이토치로 변환하는 과정을 해보자.

### 내용
*   텐서플로우 모델을 파이토치에서 사용하는 것이 불가능하지는 않다. (처음 시도해봄)
*   찾아보니 기본적으로는 파이토치에서 똑같은 모델을 코드로 짠 후, 텐서플로우 모델을 불러와 weight에 name별로 대입시켜서 파이토치 모델을 저장하여 사용한다.
*   내가 맨땅에서 짠 텐서플로우 모델을 파이토치에서 사용하려면 이런 방법을 해야하는 것 같다.◾물론 변환해주는 라이브러리들도 있지만 자세히 안찾아봄 (사용법이 그렇게 간단해보이지는 않음)
*   하지만 BERT는 매우매우 유명한 모델이고 huggingface에서 애초에 이런 변환작업을 많이 하였기 때문에 BERT 계열 모델들은 변환을 하는 가이드가 있다.◾https://huggingface.co/transformers/converting_tensorflow_models.html
*   1) Bert/GPT/GPT-2/Transformer-XL/XLNet/XLM 의 모델들을 변환할 수 있다
*   2) model.ckpt와 config.json이 필요하다.
*   3) 주어진 스크립트를 실행하면 model.bin이 생성된다.
*   4) 이 모델을 사용하기 위해서는 config.json과 vocab.txt가 필요하다.
*   model.bin을 실제 사용하는 과정◾사용할 떄는 https://huggingface.co/transformers/main_classes/model.html 의 링크를 참고하면 된다.
*   여기서 2번째 셀은 현재 주석 처리 되어있는데 이 방법은 model.bin으로 저장안하고 텐서플로우 모델을 바로 파이토치로 부르는 것이다.
*   하지만 이 방법은 로딩하는데 시간이 오래걸리고 기존의 BERT와 모델이 다른 경우는 error가 난다 (가령 vocab size가 다르다거나, hidden state 차원이 다르다거나)
*   이 모델을 사용하기 위해서는 token_idx을 넣어줘야하는데 tokenizer는 custom tokenizer이기 때문에 따로 파일을 불러읽어야 한다.
*   이 코드는 기존의 BERT 학습시킬 때 사용했던 tokenization 파일이다.
*   이제는 다음과 같이 사용하면 된다.
*   뒷 부분의 학습이나 fine-tunning은 불러 읽어들인 모델에 custom하게 코드를 짜서 사용하면 된다.
*   즉 기존의 영어 등의 pretrained_model을 사용하는 것과 같다.