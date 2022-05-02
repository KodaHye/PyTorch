# PyTorch

### 만들면서 배우는 파이토치 딥러닝

## Chapter1. 화상분류와 전이학습
[1.1 학습된 VGG모델을 사용하는 방법](https://github.com/KodaHye/PyTorch/blob/main/Chapter1.%20%ED%99%94%EC%83%81%20%EB%B6%84%EB%A5%98%EC%99%80%20%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5(VGG)/1.1%20%ED%95%99%EC%8A%B5%EB%90%9C%20VGG%20%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94%20%EB%B0%A9%EB%B2%95.ipynb) <br>
[1.2 파이토치를 활용한 딥러닝 구현 흐름](https://github.com/KodaHye/PyTorch/blob/main/Chapter1.%20%ED%99%94%EC%83%81%20%EB%B6%84%EB%A5%98%EC%99%80%20%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5(VGG)/1.2%20%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EB%94%A5%EB%9F%AC%EB%8B%9D%20%EA%B5%AC%ED%98%84%20%ED%9D%90%EB%A6%84.ipynb) <br>
[1.3 전이학습 구현](https://github.com/KodaHye/PyTorch/blob/main/Chapter1.%20%ED%99%94%EC%83%81%20%EB%B6%84%EB%A5%98%EC%99%80%20%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5(VGG)/1.3%20%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5%20%EA%B5%AC%ED%98%84.ipynb)<br>
1.4 아마존 AWS의 클라우드 GPU 머신을 사용하는 방법 <br>
[1.5 파인튜닝 구현](https://github.com/KodaHye/PyTorch/blob/main/Chapter1.%20%ED%99%94%EC%83%81%20%EB%B6%84%EB%A5%98%EC%99%80%20%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5(VGG)/1.5%20%ED%8C%8C%EC%9D%B8%ED%8A%9C%EB%8B%9D%20%EA%B5%AC%ED%98%84.ipynb)<br>


## [Chapter2. 물체감지(SSD)](https://github.com/KodaHye/PyTorch/blob/main/Chapter2.%20%EB%AC%BC%EC%B2%B4%EA%B0%90%EC%A7%80(SSD)/2.%20%EB%AC%BC%EC%B2%B4%EA%B0%90%EC%A7%80(SSD).ipynb)
2.1 물체감지란<br>
2.2 데이터셋 구현<br>
2.3 데이터 로더 구현<br>
2.4 네트워크 모델 구현<br>
2.5 순전파 함수 구현<br>
2.6 손실함수 구현<br>
2.7 학습 및 검증 실시<br>
2.8 추론 실시<br>


## [Chapter3. 시맨틱 분할(PSPNet)](https://github.com/KodaHye/PyTorch/blob/main/Chapter3.%20%EC%8B%9C%EB%A7%A8%ED%8B%B1%20%EB%B6%84%ED%95%A0(PSPNet)/3.%20%EC%8B%9C%EB%A7%A8%ED%8B%B1%20%EB%B6%84%ED%95%A0(PSPNet).ipynb)
3.1 시맨틱 분할이란<br>
3.2 데이터셋과 데이터 로더 구현<br>
3.3 PSPNet 네트워크 구성 및 구현<br>
3.4 Feature 모듈 설명 및 구현(ResNet)<br>
3.5 Pyramid Pooling 모듈의 서브 네트워크 구조<br>
3.6 Decoder, AuxLoss 모듈 설명 및 구현<br>
3.7 파인튜닝을 활용한 학습 및 검증 실시<br>
3.8 시맨틱 분할 추론<br>


## [Chapter4. 자세 추정(OpenPose)](https://github.com/KodaHye/PyTorch/blob/main/Chapter4.%20%EC%9E%90%EC%84%B8%20%EC%B6%94%EC%A0%95(OpenPose)/4.%20%EC%9E%90%EC%84%B8%20%EC%B6%94%EC%A0%95(OpenPose).ipynb)
4.1 자세 추정 및 오픈포즈 개요<br>
4.2 데이터셋과 데이터 로더 구현<br>
4.3 오픈포즈 네트워크 구성 및 구현<br>
4.4 Feature 및 Stage 모듈 설명 및 구현<br>
4.5 텐서보드 X를 사용한 네트워크 시각화 기법<br>
4.6 오픈포즈 학습<br>
4.7 오픈포즈 추론<br>

## [Chapter5. GAN을 이용한 이미지 생성(DCGAN, Self-Attention GAN)](https://github.com/KodaHye/PyTorch/blob/main/Chapter5.%20GAN%EC%9D%84%20%ED%99%9C%EC%9A%A9%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EC%83%9D%EC%84%B1(DCGAN%2C%20Self-Attention%20GAN)/5.%20GAN%EC%9D%84%20%ED%99%9C%EC%9A%A9%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EC%83%9D%EC%84%B1.ipynb)
5.1 GAN을 활용한 이미지 생성 메커니즘과 DCGAN 구현<br>
5.2 DCGAN의 손실함수, 학습, 생성<br>
5.3 Self-Attention GAN의 개요<br>
5.4 Self-Attention GAN의 학습, 생성<br>

## [Chapter6. GAN을 활용한 이상 이미지 탐지(AnoGAN, Efficient GAN)](https://github.com/KodaHye/PyTorch/blob/main/Chapter6.%20GAN%EC%9D%84%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EC%9D%B4%EC%83%81%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%ED%83%90%EC%A7%80(AnoGAN%2C%20Efficient%20GAN)/6.%20GAN%EC%9D%84%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EC%9D%B4%EC%83%81%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%ED%83%90%EC%A7%80.ipynb)
6.1 GAN을 이용한 이상 이미지 탐지 메커니즘<br>
6.2 AnoGAN 구현 및 이상 탐지 실시<br>
6.3 Efficient GAN의 개요<br>
6.4 Efficient GAN 구현 및 이상 탐지 실시<br>

## Chapater7. 자연어 처리에 의한 감정 분석(Transformer)
[7.1 형태소 분석 구현](https://github.com/KodaHye/PyTorch/blob/main/Chapter7.%20%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC%EC%97%90%20%EC%9D%98%ED%95%9C%20%EA%B0%90%EC%A0%95%20%EB%B6%84%EC%84%9D(Transformer)/7-1_Tokenizer.ipynb)<br>
[7.2 torchtext를 활용한 데이터셋, 데이터 로더 구현](https://github.com/KodaHye/PyTorch/blob/main/Chapter7.%20%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC%EC%97%90%20%EC%9D%98%ED%95%9C%20%EA%B0%90%EC%A0%95%20%EB%B6%84%EC%84%9D(Transformer)/7-2_torchtext.ipynb)<br>
[7.3 단어의 벡터 표현 방식]()<br>
[7.4 word2vec, fasttext에서 학습된 모델을 사용하는 방법](https://github.com/KodaHye/PyTorch/blob/main/Chapter7.%20%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC%EC%97%90%20%EC%9D%98%ED%95%9C%20%EA%B0%90%EC%A0%95%20%EB%B6%84%EC%84%9D(Transformer)/7-4_vectorize.ipynb)<br>
[7.5 IMDb의 데이터 로더 구현](https://github.com/KodaHye/PyTorch/blob/main/Chapter7.%20%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC%EC%97%90%20%EC%9D%98%ED%95%9C%20%EA%B0%90%EC%A0%95%20%EB%B6%84%EC%84%9D(Transformer)/7-5_IMDb_Dataset_DataLoader.ipynb)<br>
[7.6 Transformer 구현(분류 작업용)](https://github.com/KodaHye/PyTorch/blob/main/Chapter7.%20%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC%EC%97%90%20%EC%9D%98%ED%95%9C%20%EA%B0%90%EC%A0%95%20%EB%B6%84%EC%84%9D(Transformer)/7-6_Transformer.ipynb)<br>
[7.7 Transformer의 학습/추론, 판단 근거의 시각화 구현](https://github.com/KodaHye/PyTorch/blob/main/Chapter7.%20%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC%EC%97%90%20%EC%9D%98%ED%95%9C%20%EA%B0%90%EC%A0%95%20%EB%B6%84%EC%84%9D(Transformer)/7-7_transformer_training_inference.ipynb)<br>

## [Chapter8. 자연어 처리를 활용한 감정 분석(BERT)](https://github.com/KodaHye/PyTorch/blob/main/Chapter8.%20%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EA%B0%90%EC%A0%95%20%EB%B6%84%EC%84%9D(BERT)/8.%20%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EA%B0%90%EC%A0%95%20%EB%B6%84%EC%84%9D.ipynb)
8.1 BERT 메커니즘 <br>
8.2 BERT 구현 <br>
8.3 BERT를 활용한 벡터 표현(bank: 은행과 bank: 강변) <br>
8.4 BERT의 학습 및 추론, 판단 근거의 시각화 구현 <br>