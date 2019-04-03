% Generating RSA, EDR, CPC from ECG record
clear;
load('Record.mat');
fs = 250; %sampling frequency
num_p = length(Record); %number of patients
for p = 1:num_p
    % load patient specific data
    ecg = Record(p).Filtered_ecg;
    RSA = Record(p).rr;
    Rpk = Record(p).Rpk;
    tRSA = (Rpk(1:end-1) + (Rpk(2:end)-Rpk(1:end-1))/2)/250;
    testedr = edr(0,ecg,Rpk/fs,fs);
    t_edr = testedr(1:length(testedr)/2); % time vector for edr
    EDR = testedr(length(testedr)/2+1:end);
    Record(p).tEDR = t_edr;
    Record(p).EDR = EDR;
    % resample EDR and RSA
    rs_edr = resample(EDR,t_edr,4,'linear');
    rs_rsa = resample(RSA,tRSA,4,'linear');
    % generate CPC
    num_image = floor((length(EDR)-1200)/120)+1;
    cpc = zeros(num_image,50,18);
    for i = 1:num_image
        for j = 1:18
            idx_start = (i-1)*120 + (j-1)*40 +1;
            idx_end = idx_start +511;
            tmp = cpsd(rs_edr(idx_start:idx_end),rs_rsa(idx_start:idx_end),512,[],[],4);
            cpc(i,:,j) = abs(tmp(1:50));
        end
    end
    Record(p).cpc = cpc;
end
save("new_Record",'Record');