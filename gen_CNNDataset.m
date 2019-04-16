clear;
load('Record_label_cpc.mat');
num_p = length(Record);
len = zeros(num_p,1);
for p = 1:num_p
%     labels = Record(p).Modified_labels;
%     labels_start = floor(Record(p).Sample_stamps(1)/250/30)+1;
%     labels_end = floor(Record(p).Sample_stamps(length(labels))/250/30)+1;
    cpc_start = floor(Record(p).tEDR(1)/30)+1;
    cpc_end = floor(Record(p).tEDR(end)/30)+1;
%     len(p) = min(cpc_end-labels_start,labels_end-cpc_start)-10;
    len(p) = length( Record(p).cpc);
end
x = zeros(sum(len),50,18); %feature map:CPC
y = ones(sum(len),1)*100; % 5min labels (voting result of 10 30s labels)
pid = zeros(sum(len),1); % patient id
C = containers.Map;
C('W')=0;C('R')=1;C('NLS')=2;C('NDS')=3;C("0")=100;
cnt=0;
for p = 1:num_p
    cpc = Record(p).cpc;
    labels = Record(p).Modified_labels;
    labels_start = floor(Record(p).Sample_stamps(1)/250/30)+1;
    labels_end = floor(Record(p).Sample_stamps(length(labels))/250/30)+1;
    cpc_start = floor(Record(p).tEDR(1)/30)+1;
    cpc_end = floor(Record(p).tEDR(end)/30)+1;    
    empty = find(cellfun('length',labels)==0);
    labels(empty) = {""};
    tmp = strings(length(Record(p).Filtered_ecg)/250/30,1);
    tmp(labels_start:labels_end) = labels;
    labels = tmp;
    
    x(cnt+1:cnt+len(p), :,:) = cpc(1:len(p),:,:);
    for l = cpc_start:cpc_end-10
        tmp = labels(l:l+9);
        label_tbl = tabulate(tmp);
        [maxCount,idx] = max(cell2mat(label_tbl(:,2)));
        y(cnt+l) = C(string(label_tbl(idx)));
    end
%     labels = string(labels);
%     if labels_start >= cpc_start
%         x(cnt+1:cnt+len(p),:,:)=cpc(labels_start-cpc_start:...
%             labels_start-cpc_start+len(p)-1,:,:);
%         for i = 1:len(p)
%             tmp = labels(i:i+9);
%             tbl = tabulate(tmp);
%             [maxCount,idx] = max(cell2mat(tbl(:,2)));
%             y(cnt+i)=C(string(tbl(idx)));
%             pid(cnt+i)=p;
%         end
%     else
%         x(cnt+1:cnt+len(p),:,:)=cpc(1:len(p));
%         for i = 1:len(p)
%             istart = cpc_start - labels_start+i;
%             tmp = labels(istart:istart+9);
%             tbl = tabulate(tmp);
%             [maxCount,idx] = max(cell2mat(tbl(:,2)));
%             y(cnt+i)=C(string(tbl(idx)));
%             pid(cnt+i)=p;
%         end
%     end
    
    cnt = cnt + len(p);
end


    