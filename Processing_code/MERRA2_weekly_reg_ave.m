% Find regional weekly averages of temperature and corresponding z-score in
% MERRA-2 data

clear
clc

file_out_base = '/projects/subseasonal_extreme/weekly_ave_test/CONUS_Regions_';

Var = 'TLML';

Years = (1980:2024);
YY = length(Years);

Region = {'W','SW','MW','SE','NE'};

RR = length(Region);

file_data = '/pscratch/tsehrma/MERRA2/Weekly_Mean/1980.nc';

lon = ncread(file_data,'lon');
MM = length(lon);
lat = ncread(file_data,'lat');
NN = length(lat);
week_start = ncread(file_data,'week');
WW = length(week_start);

LL = MM*NN;
TT = YY*WW;
Date = reshape(repmat(Years,[WW,1]),[TT,1])*100 + repmat((1:WW),[1 YY])';

coslat = cosd(repmat(lat',[MM 1]));

for yy = 1:YY
    file_data = strcat('/pscratch/tsehrma/MERRA2/Weekly_Mean/',num2str(Years(yy)),'.nc');
    Data(:,:,(yy-1)*WW+1:yy*WW) = ncread(file_data,Var);
end
Data = reshape(Data,[LL TT]);

load('MERRA2_in_reg.mat','in_reg')

area_weight = repmat(coslat(:),[1 RR]).*in_reg;
area_weight = area_weight./repmat(sum(area_weight,1),[LL 1]);

Data_reg = area_weight'*Data;

Data_reg_year = reshape(Data_reg,[RR WW YY]);
Data_reg_mean = mean(Data_reg_year,3);
Data_reg_std = std(Data_reg_year,[],3);
Data_reg_Z = (Data_reg - repmat(Data_reg_mean,[1 YY]))./repmat(Data_reg_std,[1 YY]);

Data_out = zeros(TT,2*RR);
Data_out(:,(1:2:2*RR)) = Data_reg';
Data_out(:,(2:2:2*RR)) = Data_reg_Z';
Data_out = [Date, Data_out];

header = {'Date'};
for rr = 1:RR
    header = [header, Region{rr}, strcat(Region{rr},'_Zscore')];
end

file_out = strcat(file_out_base,num2str(Years(1)),'_to_',num2str(Years(YY)),'.csv');
Cells = [header; num2cell(Data_out)];
writecell(Cells,file_out)
