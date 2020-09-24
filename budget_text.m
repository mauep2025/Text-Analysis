textData = [
 "A large tree is downed and blocking traffic outside Apple Hill."
 "There is lots of damage to many car windshields in the parking lot."];

documents = preprocessTextData(textData)
filename = "precgpe_2021.pdf";
str = extractFileText(filename);
document = tokenizedDocument(str);

% Lemmatize the words. To improve lemmatization, first use
% addPartOfSpeechDetails.
document = addPartOfSpeechDetails(document);
document = normalizeWords(document,'Style','lemma');
document = erasePunctuation(document);
document = removeShortWords(document,3);
words    = ["así" "su" "al" "el" "ju" "y" "que" "por" "una" "la" "de" "para" "como" "2018" "2019" "2020" "2021" "durante" "entre" "este" "esta" "parte" "registró" "sobre" "meses"]
document = removeWords(document, words)
figure
wordcloud(document);
title("Pre-Criterios 2021")

%%
filename_1 = "cgpe_2021.pdf";
str_1 = extractFileText(filename_1);
document_1 = tokenizedDocument(str_1);

% Lemmatize the words. To improve lemmatization, first use
% addPartOfSpeechDetails.
document_1 = addPartOfSpeechDetails(document_1);
document_1 = normalizeWords(document_1,'Style','lemma');
document_1 = erasePunctuation(document_1);
document_1 = removeShortWords(document_1,3);
words    = ["así" "travéz" "asimismo" "mismo" "otros" "cifras" "hasta" "tasa" "anterior" "respecto" "forma" "total" "términos" "además" "años" "respecto" "su" "al" "el" "ju" "y" "que" "por" "una" "la" "de" "para" "como" "cual" "ciento" "2018" "2008" "2010" "2009" "2019" "2020" "2021" "durante" "entre" "este" "esta" "parte" "registró" "sobre" "meses"]
words_1  = ["pandemia" "crisis" "influenza" "A/H1N1" "covid" "covid-19"]
document_1 = removeWords(document_1, words)
bag_1 = bagOfWords(document_1)
tbl = topkwords(bag_1,10)
figure
wordcloud(document_1);
title('\fontsize{24}CGPE 2021')

%%

filename_2 = "cgpe_2010.pdf";
str_2 = extractFileText(filename_2);
document_2 = tokenizedDocument(str_2);

% Lemmatize the words. To improve lemmatization, first use
% addPartOfSpeechDetails.
document_2 = addPartOfSpeechDetails(document_2);
document_2 = normalizeWords(document_2,'Style','lemma');
document_2 = erasePunctuation(document_2);
document_2 = removeShortWords(document_2,3);
words    = ["así" "travéz" "asimismo" "otros" "cifras" "ello" "hasta" "tasa" "anterior" "respecto" "total" "términos" "además" "años" "respecto" "su" "al" "el" "ju" "y" "que" "por" "una" "la" "de" "para" "como" "cual" "marco" "variación" "cierre" "ciento" "2003" "2011" "2012" "20102015" "2007" "2006" "20072008" "2018" "2008" "2010" "2009" "2019" "2020" "2021" "durante" "entre" "este" "esta" "parte" "registró" "sobre" "meses"]
words_1  = ["pandemia" "crisis" "influenza" "A/H1N1" "covid" "covid-19"]
document_2 = removeWords(document_2, words)
bag_2 = bagOfWords(document_2)
tbl = topkwords(bag_2,10)
similarities = bm25Similarity(document_2);
T = wordCloudCounts(str_2);


figure
wordcloud(document_2);
title('\fontsize{24}CGPE 2010')

figure
wordcloud(bag);
title('\fontsize{24}CGPE 2010')

%%
%-----------Criterios Generales 2010------------------%
clear; 
clc;
filename_2 = "cgpe_2010.txt";
emb = trainWordEmbedding(filename_2)
enc = wordEncoding(document_2)
%filename_2 = "cgpe_2010.pdf";
%str_2 = extractFileText(filename_2);
%document_2 = tokenizedDocument(str_2);
%emb_1 = trainWordEmbedding(document_2)
%sequences = doc2sequence(emb,document_2,'PaddingDirection','none');


data_1 = readtext;
idx = data_1.Label == "Positivo";
head(data_1(idx,:))
idx = data_1.Label == "Negativo";
head(data_1(idx,:))
%idx = ~isVocabularyWord(emb,data_1.Word);
idx = ~isVocabularyWord(enc,data_1.Word)
%idx_1 = ~isVocabularyWord(emb,document_2.Vocabulary);
%words = document_2.Vocabulary;
data_1(idx,:) = [];

%pos = data_1(1:59,1);
%pos = data_1(60:138,1);


numWords = size(data_1,1);
%cvp = cvpartition(numWords,'HoldOut');
cvp = cvpartition(numWords,'HoldOut',0.1);
dataTrain = data_1(training(cvp),:);	
dataTest = data_1(test(cvp),:);

wordsTrain = dataTrain.Word;
wordsc     =data_1.Word;
Xc     = word2vec(emb,wordsc);
XTrain = word2vec(emb,wordsTrain);
%XTrain_1 = word2vec(enc,wordsTrain);
Yc     = data_1.Label;
YTrain = dataTrain.Label;


%words = documents.Vocabulary;
%words(ismember(words,wordsTrain)) = [];

mdl = fitcsvm(XTrain,YTrain);
mdl = fitcsvm(Xc,Yc);
mdl.Prior %firts positive and then negative
%mdl = fitcnb(XTrain,YTrain);
wordsTest = dataTest.Word;

XTest = word2vec(emb,wordsTest);
YTest = dataTest.Label;

[YPred,scores] = predict(mdl,XTest);
[YPred,scores] = predict(mdl,Xc);

figure
confusionchart(YTest,YPred);
title('Criterios 2010')



figure
subplot(1,2,1)
idx = YPred == "Positivo";
wordcloud(wordsTest(idx),scores(idx,1));
title("Predicted Positive Sentiment")

subplot(1,2,2)
wordcloud(wordsTest(~idx),scores(~idx,2));
title("Predicted Negative Sentiment")


%////////////////
idx_2=(data_1.Label=="Positivo")
s_2=sum(idx_2(:) == 1);
idx_3=(data_1.Label=="Negativo")
s_3=sum(idx_3(:) == 1);

w_cloud_p=data_1.Word(1:s_2,:)
w_cloud_n=data_1.Word(s_2:end,:)

figure
subplot(1,2,1)
%idx = "Positivo";
wordcloud(w_cloud_p);
title('\fontsize{24}Positivas')

subplot(1,2,2)
wordcloud(w_cloud_n);
title('\fontsize{24}Negativas')



%%
%-----------Criterios Generales 2021------------------%
clear; 
clc;
filename_2 = "cgpe_2021.txt";
emb = trainWordEmbedding(filename_2)
data_1 = readtext;
idx = data_1.Label == "Positivo";
head(data_1(idx,:))
idx = data_1.Label == "Negativo";
head(data_1(idx,:))
idx = ~isVocabularyWord(emb,data_1.Word);
data_1(idx,:) = [];

numWords = size(data_1,1);
cvp = cvpartition(numWords,'HoldOut',0.9);
dataTrain = data_1(training(cvp),:);
dataTest = data_1(test(cvp),:);

wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;
%mdl = fitcnb(XTrain,YTrain);
mdl = fitcsvm(XTrain,YTrain);
mdl.Prior %firts positive and then negative
wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest);
YTest = dataTest.Label;

[YPred,scores] = predict(mdl,XTest);

figure
confusionchart(YTest,YPred);
title('Criterios 2021')


figure
subplot(1,2,1)
idx = YPred == "Positivo";
wordcloud(wordsTest(idx),scores(idx,1));
title("Predicted Positive Sentiment")

subplot(1,2,2)
wordcloud(wordsTest(~idx),scores(~idx,2));
title("Predicted Negative Sentiment")




%%
emb = fastTextWordEmbedding;
data = readLexicon;
idx = data.Label == "Positive";
head(data(idx,:))
idx = data.Label == "Negative";
head(data(idx,:))
idx = ~isVocabularyWord(emb,data.Word);
data(idx,:) = [];

numWords = size(data,1);
cvp = cvpartition(numWords,'HoldOut',0.1);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);

wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

mdl = fitcsvm(XTrain,YTrain);
wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest);
YTest = dataTest.Label;

[YPred,scores] = predict(mdl,XTest);

figure
confusionchart(YTest,YPred);


figure
subplot(1,2,1)
idx = YPred == "Positive";
wordcloud(wordsTest(idx),scores(idx,1));
title("Predicted Positive Sentiment")

subplot(1,2,2)
wordcloud(wordsTest(~idx),scores(~idx,2));
title("Predicted Negative Sentiment")



