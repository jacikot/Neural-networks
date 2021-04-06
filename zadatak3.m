clc,clear,close all

data=readmatrix('CO2_dataset.csv');

data=data';
N=length(data);
entry=data(1:9,:);
exit=data(10,:);


%% Traing and test separation
rIndex=randperm(N);

Nt=floor(0.85*N);
trainInd=rIndex(1:Nt);
testInd=rIndex(Nt+1:N);
valInd=rIndex(floor(Nt*0.9)+1:Nt);
entryTrain=entry(:,trainInd);
exitTrain=exit(:,trainInd);
entryTest=entry(:,testInd);
exitTest=exit(:,testInd);
entryVal=entry(:,valInd);
exitVal=exit(:,valInd);

%% Crossvalidation

best_perf=-1;
arc{1}=[20,25,30,25];
arc{2}=[25,30,30];
arc{3}=[28,22,18,19];
arc{4}=[25,19,35];
for j=1:length(arc)

    for fcn=["logsig","poslin","softmax","tansig"]
    
        for reg=[0.3,0.5,0.75,0.9]
            
           net=fitnet(arc{j});
            for i=1:length(arc{j})
                net.layers{i}.transferFcn=fcn;
            end
             net.divideFcn='divideind';
             net.divideParam.trainInd=1:floor(0.9*Nt);
             net.divideParam.valInd=floor(0.9*Nt)+1:Nt;
             net.divideParam.testInd=[];
           
            net.trainFcn='trainlm';
            net.performParam.regularization=reg;
            net.trainParam.epochs=1000;
            net.trainParam.goal=1e-5;
            net.trainParam.min_grad=1e-5;
            net.trainParam.max_fail=10;
            net.trainParam.showWindow=false;
            net.trainParam.showCommandLine=true;
            
            [net,tr]=train(net,entryTrain,exitTrain);
         
            
            perf=perform(net,entryVal,exitVal);
            
            if (best_perf==-1) || (best_perf >perf)
                best_perf=perf;
                best_arc=arc{j};
                best_fcn=fcn;
                best_reg=reg;
                best_ep=tr.best_epoch;
            end
            
       end
    end
    
end
    
%% Training with best params

net=fitnet(best_arc);
for i=1:length(best_arc)
     net.layers{i}.transferFcn=best_fcn;
end
   net.divideFcn='';
   net.trainFcn='trainlm';
   net.performParam.regularization=best_reg;
   net.trainParam.epochs=best_ep;
   net.trainParam.goal=1e-5;
   net.trainParam.min_grad=1e-5;
   net.trainParam.max_fail=10;
   net.trainParam.showWindow=false;
    net.trainParam.showCommandLine=true;
   [net,tr]=train(net,entryTrain,exitTrain);
   figure,
    plot(tr.perf);
   
%% Testing 

predTest=net(entryTest);
predTrain=net(entryTrain);
figure('Name','Regresiona kriva trening'),hold all
plotregression(exitTrain,predTrain);
figure('Name','Regresiona kriva'),hold all
plotregression(exitTest,predTest);

