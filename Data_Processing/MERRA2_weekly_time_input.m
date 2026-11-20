% Timeseries input file for weekly MERRA-2 data forecast

clear
clc

file_out_base = '/projects/subseasonal_extreme/weekly_ave_test/time_inputs_';

Years = (1980:2024);
YY = length(Years);

Period = [10 5 2 1 0.5 0.25];
PP = length(Period);

file_data = '/pscratch/tsehrma/MERRA2/Weekly_Mean/1980.nc';
WW = length(ncread(file_data,'week'));

TT = YY*WW;
Date = reshape(repmat(Years,[WW,1]),[TT,1])*100 + repmat((1:WW),[1 YY])';

week = (1:TT)';

trig_in = (2*pi/WW)*week;

Data_trig = zeros(TT,2*PP);

header = {'Date','Week'};

for pp = 1:PP
    Data_trig(:,2*pp-1) = cos(trig_in/Period(pp));
    Data_trig(:,2*pp) = sin(trig_in/Period(pp));
    header = [header, strcat(num2str(Period(pp)),'yr_cos'), strcat(num2str(Period(pp)),'yr_sin')];
end

Data_out = [Date, week, Data_trig];

file_out = strcat(file_out_base,num2str(Years(1)),'_to_',num2str(Years(YY)),'.csv');
Cells = [header; num2cell(Data_out)];
writecell(Cells,file_out)
