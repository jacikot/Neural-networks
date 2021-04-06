clc,clear,close all

% Finding the needed scope of x and setting the starting values
N=1500;
x=linspace(-1,1,N);
h=5*sin(40*pi*x)+4*sin(12*pi*x);
std=0.2*4;
y=h+std*randn(1,N);


%% Plotting h(x) and y(x) / prva tacka

figure(),hold all;
plot(x,h,'LineWidth',1);
plot(x,y);
legend('h(x)','y(x)');

%% Creating Neural Network / druga stavka

net=fitnet([10 12 7]);
%net.layers{1}.transferFcn='logsig';
net.divideFcn='';
net.trainParam.epochs=1500;
net.trainParam.goal=1e-5;
net.trainParam.min_grad=1e-5;

%% Training of the Neural Network / treca stavka

index=randperm(N);
indexTrain=index(:,1:0.85*N);
indexTest=index(:,0.85*N+1:N);

entryTrain=x(indexTrain);
exitTrain=y(indexTrain);

entryTest=x(indexTest);
exitTest=y(indexTest);


net=train(net,entryTrain,exitTrain);

%% Simulation of the Neural Network


simulatedValues=net(entryTest);


figure, hold all
plot(x,net(x),'LineWidth',1);
plot(x,y);
legend('predikcija NM','y(x)');
