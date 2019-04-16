load('Record_label_cpc.mat');
num_p = length(Record);
C = containers.Map;
C('W')=0;C('R')=1;C('NLS')=2;C('NDS')=3;C("0")=100;
tbl = AllPatientsHRVresultsallwindows20190413;
x = zeros(size(tbl,1)-18,size(tbl,2)-3);
y = ones(size(tbl,1)-18,1)*5;
cnt = 0;

for p=1:num_p
    labels = Record(p).Modified_labels;
    empty = find(cellfun('length',labels)==0);
    labels(empty) = {""};
    labels = string(labels);
    labels_start = floor(Record(p).Sample_stamps(1)/250/30)+1;
    labels_end = floor(Record(p).Sample_stamps(length(labels))/250/30)+1;
    cpc_start = floor(Record(p).tEDR(1)/30)+1;
    cpc_end = length(Record(p).cpc)+cpc_start + 9;
    tmp = strings(length(Record(p).Filtered_ecg)/250/30,1);
    tmp(labels_start:labels_end) = labels;
    labels = tmp;
    for l=1:cpc_end-cpc_start-9
        x(cnt+l, :) = table2array(tbl(cnt+l,4:end));
        tmp = labels(l+cpc_start-1:l+9+cpc_start-1);
        label_tbl = tabulate(tmp);
        [maxCount,idx] = max(cell2mat(label_tbl(:,2)));
        y(cnt+l) = C(string(label_tbl(idx)));
    end
    cnt = cnt - cpc_start + cpc_end - 9;
end
x(y==100,:)=[];
y(y==100)=[];
x(y==5,:)=[];
y(y==5)=[];