%% Reading data
clc,clear,close all

load 'dataset1.mat'

pod=pod';
N=length(pod);
exit=pod(3,:);
entry=pod(1:2,:);
classOne=pod(1:2,exit==1);
classTwo=pod(1:2,exit==2);
classThree=pod(1:2,exit==3);

figure, hold all
plot(classOne(1,:),classOne(2,:),'bo');
plot(classTwo(1,:),classTwo(2,:),'r*');
plot(classThree(1,:),classThree(2,:),'gx');
legend('Prva klasa','Druga Klasa','Treca Klasa');

%% Test and traing set separation

% Ukoliko bismo uzeli npr. samo 70% bez prethodnog mesanja desilo bi se da
% uzimamo samo prvu i drugu klasu, dok trecu ne uzimamo. Zbog toga bi nam
% test skup imao samo trecu klasu. Treba da imamo ravnopravan broj svake
% klase.- zasto se mesaju podaci
% Test skup sluzi za utvrdjivanje koliko je nasa neuralna mreza zapravo
% dobro obucena -  jer su to novi podaci koje nikad nije videla. Trening
% skup sluzi za obucavanje mreze, dok test sluzi za proveru.

newExit=zeros(3,N);
newExit(1,exit==1)=1;
newExit(2,exit==2)=1;
newExit(3,exit==3)=1;

index=randperm(N);
indexEntryTrain=index(1:0.85*N);
indexEntryTest=index(0.85*N+1:N);

entryTrain=entry(:,indexEntryTrain);
exitTrain=newExit(:,indexEntryTrain);

entryTest=entry(:,indexEntryTest);
exitTest=newExit(:,indexEntryTest);

%% Neural Network Creation
% optimalArchitecture=[6,4];
optimalArchitecture=[4,4,3];
underfittingArchitecture=[2,3];
overfittingArchitecture=[40,50,40,30,30];

netOptimal=patternnet(optimalArchitecture);
netUnderfitting=patternnet(underfittingArchitecture);
netOverfitting=patternnet(overfittingArchitecture);

 

%% Optimal neural network parameters 

netOptimal.divideFcn='';
netOptimal.trainParam.epochs=600;
netOptimal.trainParam.min_grad=1e-5;
netOptimal.trainParam.goal=1e-5; 

%% Underfitting neural network parameters

netUnderfitting.divideFcn='';
netUnderfitting.trainParam.epochs=600;
netUnderfitting.trainParam.min_grad=1e-5;
netUnderfitting.trainParam.goal=1e-5; 

%% Overfitting neural network parameters

netOverfitting.divideFcn='';
netOverfitting.trainParam.epochs=600;
netOverfitting.trainParam.min_grad=1e-5;
netOverfitting.trainParam.goal=1e-5; 


%% Training of Neural Networks

[netOptimal,trOptimal]=train(netOptimal,entryTrain,exitTrain);
[netUnderfitting,trUnderfitting]=train(netUnderfitting,entryTrain,exitTrain);
[netOverfitting,trOverfitting]=train(netOverfitting,entryTrain,exitTrain);

figure('Name','Optimal performance'),hold all
plot(trOptimal.perf);

figure('Name','Underfitting performance'), hold all
plot(trUnderfitting.perf);

figure('Name','Overfitting perfomance'), hold all
plot(trOverfitting.perf);

%% Testing neural Networks

predTrainOptimal=netOptimal(entryTrain);
predTestOptimal=netOptimal(entryTest);

predTrainUnder=netUnderfitting(entryTrain);
predTestUnder=netUnderfitting(entryTest);

predTrainOver=netOverfitting(entryTrain);
predTestOver=netOverfitting(entryTest);

%optimal
figure
plotconfusion(exitTrain,predTrainOptimal,'Train Optimal');
figure
plotconfusion(exitTest,predTestOptimal,'Test Optimal');

%under
figure 
plotconfusion(exitTrain,predTrainUnder,'Train Under');
figure
plotconfusion(exitTest,predTestUnder,'Test Under');

%over
figure 
plotconfusion(exitTrain,predTrainOver,'Train Over');
figure
plotconfusion(exitTest,predTestOver,'Test Over');

[~,cmOptimal]=confusion(exitTest,predTestOptimal);
[~,cmUnder]=confusion(exitTest,predTestUnder);
[~,cmOver]=confusion(exitTest,predTestOver);

cmOptimal=cmOptimal';
cmUnder=cmUnder';
cmOver=cmOver';

Poptimal=cmOptimal(1,1)/sum(cmOptimal(1,:));
Roptimal=cmOptimal(1,1)/sum(cmOptimal(:,1));

Punder=cmUnder(1,1)/sum(cmUnder(1,:));
Runder=cmUnder(1,1)/sum(cmUnder(:,1));

Pover=cmOver(1,1)/sum(cmOver(1,:));
Rover=cmOver(1,1)/sum(cmOver(:,1));

%% Decision border 

Dots=500;
repMatrix1=repmat(linspace(-5,5,Dots),1,Dots);
repMatrix2=repelem(linspace(-5,5,Dots),Dots);

entryPaint=[repMatrix1;repMatrix2];

predPaintOptimal=netOptimal(entryPaint);
predPaintUnder=netUnderfitting(entryPaint);
predPaintOver=netOverfitting(entryPaint);

[~,exitPaintOptimal]=max(predPaintOptimal);
[~,exitPaintUnder]=max(predPaintUnder);
[~,exitPaintOver]=max(predPaintOver);

%Optimal 
classOneOptimal=entryPaint(:,exitPaintOptimal==1);
classTwoOptimal=entryPaint(:,exitPaintOptimal==2);
classThreeOptimal=entryPaint(:,exitPaintOptimal==3);


figure('Name','Optimal'),hold all
plot(classOneOptimal(1,:),classOneOptimal(2,:),'.');
plot(classTwoOptimal(1,:),classTwoOptimal(2,:),'.');
plot(classThreeOptimal(1,:),classThreeOptimal(2,:),'y.');
plot(classOne(1,:),classOne(2,:),'bo');
plot(classTwo(1,:),classTwo(2,:),'r*');
plot(classThree(1,:),classThree(2,:),'gx');

%Under
classOneUnder=entryPaint(:,exitPaintUnder==1);
classTwoUnder=entryPaint(:,exitPaintUnder==2);
classThreeUnder=entryPaint(:,exitPaintUnder==3);


figure('Name','Underfitting'),hold all
plot(classOneUnder(1,:),classOneUnder(2,:),'.');
plot(classTwoUnder(1,:),classTwoUnder(2,:),'.');
plot(classThreeUnder(1,:),classThreeUnder(2,:),'y.');
plot(classOne(1,:),classOne(2,:),'bo');
plot(classTwo(1,:),classTwo(2,:),'r*');
plot(classThree(1,:),classThree(2,:),'gx');


%Over
classOneOver=entryPaint(:,exitPaintOver==1);
classTwoOver=entryPaint(:,exitPaintOver==2);
classThreeOver=entryPaint(:,exitPaintOver==3);


figure('Name','Overfitting'),hold all
plot(classOneOver(1,:),classOneOver(2,:),'.');
plot(classTwoOver(1,:),classTwoOver(2,:),'.');
plot(classThreeOver(1,:),classThreeOver(2,:),'y.');
plot(classOne(1,:),classOne(2,:),'bo');
plot(classTwo(1,:),classTwo(2,:),'r*');
plot(classThree(1,:),classThree(2,:),'gx');
