
folder='./';

files=dir([folder '*iplot.txt']);
for iF=1:numel(files)

    fn=files(iF).name; 
    fn2=strrep(fn,'iplot.txt','iplot.dens.txt');
    if ~exist(fullfile(folder, fn2),'file')
        warning(['Missing file ' fn2 ' for file ' fn '. Skipping ' fn '.']);
        continue
    end
    
    f=fopen(fullfile(folder, fn), 'r');
    s=fgets(f);
    s=fgets(f);
    var1 = strtrim(s(8:end));
    s=fgets(f);
    var2 = strtrim(s(11:end));
    if index(fn, "chosen") == 1
        legendtxt1 = fgets(f);
        legendtxt2 = fgets(f);
        fclose(f);
        [data]=dlmread(fullfile(folder, fn),'\t',7,0);
    else
        fclose(f);
        [data]=dlmread(fullfile(folder, fn),'\t',5,0);
    end

    xvalues = data(2, 3:end)';
    xcounts = data(1, 3:end)';
    yvalues = data(3:end, 2);
    ycounts = data(3:end, 1);
    values = data(3:end, 3:end);

    [density]=dlmread(fullfile(folder, fn2),'\t',4,0);

    if index(fn, "chosen") == 1
        make_interaction_plot_lrtree(xvalues, xcounts, yvalues, ycounts, values, density, var2, fn, [legendtxt1; legendtxt2]);
        print(gcf,'-depsc',[folder fn '.eps']);
    else
    
        make_interaction_plot_lrtree(xvalues, xcounts, yvalues, ycounts, values, density, var2, fn);
        print(gcf,'-depsc',[folder fn '.eps']);

        make_interaction_plot_lrtree(yvalues, ycounts, xvalues, xcounts, values', density', var1, ['Flipped ' fn]);
        print(gcf,'-depsc',[folder fn '.flipped.eps']);
    end

end