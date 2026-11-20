% Create weekly mean data from MERRA2 data

clear
clc

Var = 'TLML';%'SLP';%'TS';%'TLML'

exp_name = 'tavg1_2d_flx_Nx'; % TLML
% exp_name = 'tavg1_2d_rad_Nx'; % TS
% exp_name = 'inst3_3d_asm_Np'; %SLP
base = strcat('/gpfs/cldera/data/MERRA-2/',exp_name,'/');

Years = (1980:2024);
YY = length(Years);

WL = 7;
Week_start = (1:WL:360);
WW = length(Week_start);

file_name = strcat(base,'1980/01/MERRA2_100.',exp_name,'.19800101.nc4');
lon = ncread(file_name,'lon');
II = length(lon);
lat = ncread(file_name,'lat');
JJ = length(lat);

for yy = 1:YY

    year = Years(yy);
    disp(year)

    month_day = [31 28+(mod(year,4)==0) 31 30 31 30 31 31 30 31 30 31];
    DD = sum(month_day);
    Months = ones(DD,1);
    Days = (1:DD)';
    for dd = 1:DD
        while Days(dd) > month_day(Months(dd))
            Days(dd) = Days(dd) - month_day(Months(dd));
            Months(dd) = Months(dd) + 1;
        end
    end
    Week_start(WW+1) = DD+1;

    ww = 1;
    wl = 1;
    Data_day = zeros(II,JJ,WL);
    Data_week = zeros(II,JJ,WW);

    for dd = 1:DD
        month = Months(dd);
        day = Days(dd);

        if year<1992
            f_num = '100';
        elseif year<2001
            f_num = '200';
        elseif year<2011
            f_num = '300';
        elseif (year==2020 && month==9) || (year==2021 && (month>=6 && month<=9))
            f_num = '401';
        else
            f_num = '400';
        end
        file_name = strcat(base,num2str(year),'/',num2str(month,'%02d'),'/MERRA2_',f_num,'.',exp_name,'.',num2str(year),num2str(month,'%02d'),num2str(day,'%02d'),'.nc4');

        if dd == Week_start(ww+1)
            Data_week(:,:,ww) = mean(Data_day,3);
            Data_day = zeros(II,JJ,WL);
            wl = 1;
            ww = ww + 1;
        end

        Data_day(:,:,wl) = mean(ncread(file_name,Var),3);
        wl = wl + 1;
    end
    Data_week(:,:,WW) = mean(Data_day,3);

    file_out = strcat('/pscratch/tsehrma/MERRA2/Weekly_Mean/',num2str(year),'.nc');

    if exist(file_out,'file') == 0
        nccreate(file_out,'lon','Dimensions',{'lon',II})
        nccreate(file_out,'lat','Dimensions',{'lat',JJ})
        nccreate(file_out,'week','Dimensions',{'week',WW})
        ncwrite(file_out,'lon',lon)
        ncwrite(file_out,'lat',lat)
        ncwrite(file_out,'week',Week_start(1:WW))
    end

    nccreate(file_out,Var,'Dimensions',{'lon',II,'lat',JJ,'week',WW})
    ncwrite(file_out,Var,Data_week)
end

exit
