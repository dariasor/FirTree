
function make_effect_plot(xvalues, xcounts, values, plot_title)

clf
axes('position',[0.1 0.2,0.8 0.7])
hold on

%lc = [198 198 198]/255;
%lw = 3;
h = plot(xvalues, values);
set(h,'LineWidth', 5);
set(h,'Color', [127 127 127]/255)
    
xLegend=cell(1, numel(xvalues));
for iX=1:numel(xvalues)
    if xcounts(iX) == 1
        s = num2str(xvalues(iX));
    else
        s = [num2str(xvalues(iX)) ' (x ' num2str(xcounts(iX)) ')'];
    end
    xLegend{iX} = s;
end
set(gca, 'XTick', xvalues, 'XTickLabel', xLegend)
rotateticklabel(gca,45);
title(fix_title(plot_title)) 