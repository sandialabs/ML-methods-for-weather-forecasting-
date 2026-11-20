% regional inputs of MERRA2 weekly averaged data including aves and PCs

clear
clc

PC_num = 20;

Var = {'TS','SLP','T_850','T_500','H_850','H_500'};
VV = length(Var);

region_in = {'conus','southern_canada','mexico_gulf','east_asia','atlantic_ocean','pacific_ocean'};
RRR = length(region_in);

region = {'conus','southern_canada','mexico_gulf','east_asia','atlantic_ocean','pacific_ocean','atlantic_trop','pacific_trop','arctic'};
RR = length(region);

base_in = '/pscratch/tsehrma/Seasonal_Forecast/region_map/';
base_out = '/projects/subseasonal_extreme/weekly_ave_test/additional_inputs/';
base_data = '/pscratch/tsehrma/MERRA2/Weekly_Mean/';
file_E_var = '/projects/subseasonal_extreme/weekly_ave_test/input_explained_variance.csv';
file_E_var_cu = '/projects/subseasonal_extreme/weekly_ave_test/input_explained_variance_cumulative.csv';

Years = (1980:2024);
YY = length(Years);

file_data = strcat(base_data,'1980.nc');
lon = ncread(file_data,'lon');
MM = length(lon);
lat = ncread(file_data,'lat');
NN = length(lat);
week_start = ncread(file_data,'week');
WW = length(week_start);

TT = YY*WW;
Date = reshape(repmat(Years,[WW,1]),[TT,1])*100 + repmat((1:WW),[1 YY])';

LL = MM*NN;
lat_full = repmat(lat',[MM 1]);
lon_full = repmat(lon,[1 NN]);
coslat = cosd(lat_full(:));

Years_train = (1980:2016);
TTT = length(Years_train)*WW;

glob_reg = false(LL,RR);

for rr = 1:RRR
    file_in = strcat(base_in,region_in{rr},'.csv');
    region_coord = readmatrix(file_in);

    [in,on] = inpolygon(lon_full(:),lat_full(:),region_coord(:,2),region_coord(:,1));
    if rr == RRR
        glob_reg(:,rr) = and(and(lat_full(:)>=-20,lat_full(:)<60),~in);
    else
        glob_reg(:,rr) = or(in,on);
    end
end

glob_reg(:,RRR+1:RRR+2) = and(glob_reg(:,RRR-1:RRR),repmat(lat_full(:)<=20,[1 2]));
glob_reg(:,RRR-1:RRR) = and(glob_reg(:,RRR-1:RRR),~glob_reg(:,RRR+1:RRR+2));
glob_reg(:,RR) = lat_full(:)>=60;

%%

headers_E_var = {'PC_num'};
E_var = zeros(PC_num,VV*RR);
E_var_cu = zeros(PC_num,VV*RR);

headers = {'Date','Mean'};
for pp = 1:PC_num
    headers = [headers, strcat('PC',num2str(pp))];
end

% vv = 7;
for vv = 1:VV

    disp(Var{vv})

    Data_all = zeros(LL,TT);

    for yy = 1:YY
        file_data = strcat(base_data,num2str(Years(yy)),'.nc');
        Data_all(:,(yy-1)*WW+1:yy*WW) = reshape(ncread(file_data,Var{vv}),[LL WW]);
    end

    Data_clim = mean(reshape(Data_all,[LL WW YY]),3);
    Data_anom = Data_all - repmat(Data_clim,[1 YY]);

    % rr = 1;
    for rr = 1:RR

        disp(region{rr})

        file_out = strcat(base_out,Var{vv},'_weekave_',region{rr},'_PC',num2str(PC_num),'_',num2str(Years(1)),'_to_',num2str(Years(YY)),'.csv');

        Data_reg = Data_anom(glob_reg(:,rr),:);
        area_weight = coslat(glob_reg(:,rr))';

        nan_check = isnan(sum(Data_reg,2));
        if sum(nan_check) ~= 0
            Data_reg(nan_check,:) = [];
            area_weight(nan_check) = [];
        end
        area_weight = area_weight/sum(area_weight);

        Data_reg_mean = area_weight*Data_reg;

        Data_reg_use = Data_reg(:,1:TTT) - repmat(mean(Data_reg(:,1:TTT),2),[1 TTT]);
        Co_Var = (1/TT)*(Data_reg_use*Data_reg_use');

        [EOF,D] = eigs(Co_Var,PC_num);
        E_var(:,(vv-1)*RR+rr) = diag(D)/sum(diag(Co_Var));
        for pp = 1:PC_num
            E_var_cu(pp,(vv-1)*RR+rr) = sum(E_var(1:pp));
        end

        headers_E_var = [headers_E_var,strcat(Var{vv},'_',region{rr})];

        PC = Data_reg'*EOF;

        Data = [Date,Data_reg_mean',PC];
        Cells = [headers; num2cell(Data)];
        writecell(Cells,file_out)

        if exist(file_out,'file') ~= 0
            disp('file written')
        end
    end
end

writecell([headers_E_var; num2cell([(1:PC_num)',E_var])],file_E_var)
writecell([headers_E_var; num2cell([(1:PC_num)',E_var_cu])],file_E_var_cu)

exit
