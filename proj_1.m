
%% pre processing data

clc
clear


data_folder = 'D:\6254 stat ML\proj\Data_proj\';    % Folder containing Ensemble Averaged Data
data_files = fullfile(data_folder, '*.mat');          % Reading mat files for Eavg data
dataFiles = dir(data_files);

% Initialize settings 
HRVparams = InitializeHRVparams('Proj_6254');  
HRVparams.writedata = 'Proj_6254';            % parameters initializain
HRVparams.Fs = 250;
HRVparams.ext =  'mat';

Record = struct();                    

%%
for k = 1:length(dataFiles)
  
  % 1. Loading data
  basename_data = dataFiles(k).name;                         % Reading the ECG /ICG record
  full_data = fullfile(data_folder, basename_data);
  fprintf(1,'Loading %s\n', basename_data);
  load(full_data);
  [~, aa2] = find(basename_data=='.');
  Record(k).ID = basename_data(1:aa2-1);
  
  % 2. FIR Filtering of data
   lpf = lpf_proj;
   hpf = hpf_proj;
   val_lp = filter(lpf,val);
   ecg = filter(hpf,val_lp);
   Record(k).Filtered_ecg = ecg;

  
  % 3. Extracting R peaks and RR intervals
  [Rpeaks,sign,en_thres] = jqrs(ecg,HRVparams);     
  rr = diff(Rpeaks)./HRVparams.Fs; 
  Record(k).Rpk = Rpeaks;
  Record(k).rr = rr;
  t = cumsum(rr);
  
 end
%%
save('Record.mat','Record')  % Save the structure
