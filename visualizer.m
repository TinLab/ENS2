% version 20220518
clear;
close all;
clc;

file_name = 'sample_data';

data_name = 'dff';

% [for saving figures and data]
is_saving = false;

%%
load(['./results/', file_name, '_ENS2.mat']);
data = eval([data_name,'_ENS2']); 

global param
param = struct();
param.trial_num       = 20;
param.trial_duration  = 10;
param.frame_rate      = 10;
param.sampling_rate   = 60;
param.stim_onset      = 6;
param.orient_num      = 8;
param.orient_duration = 0.5;

%%
win_size = get(0, 'Screensize' );

for neuron = 1:length(data)
    
    % skip unresponsive neuron
%     if nansum(data{neuron}.calcium) == 0
%         continue
%     end
    
%     In each cell:
%     neuron:                neuron number
%     frame_rate:            original imaging frequency
%     fluo_times:            timestamp of raw dff
%     raw_dff:               raw calcium trace
%     dff_resampled:         calcium trace resampled at 60Hz
%     dff_resampled_segment: segmented trace (input of ENS2, not provided)
%     fluo_times_resampled:  timestamp of resampled calcium trace
%     frame_rate_resampled:  resampling frequency
%     calcium:               same as dff_resampled
%     pd_rate:               spike rate prediction from ENS2 (probabilities)
%     pd_spike:              spike count prediction from ENS2 (spike trains)
%     pd_event:              spike event prediction from ENS2 (timestamps)
    

    time        = reshape_data(data{neuron}.fluo_times_resampled, param.sampling_rate);
    calcium     = reshape_data(data{neuron}.calcium, param.sampling_rate);
    spike       = reshape_data(data{neuron}.pd_spike, param.sampling_rate);
    rate        = reshape_data(data{neuron}.pd_rate, param.sampling_rate);
    raw_time    = reshape_data(data{neuron}.fluo_times, param.frame_rate);
    raw_calcium = reshape_data(data{neuron}.raw_dff, param.frame_rate);
    

    figure('position', [win_size(1)+200, win_size(2)+50, 840, 450]);
    t = tiledlayout(4,2,'TileSpacing','tight');
    
    nexttile(1,[1 2]);%h1 = subplot(10,4,1:4);
    plot(raw_time(1,:), mean(raw_calcium,1), 'color', [0.2, 0.2, 0.2], 'linewidth', 1.5);
    xlim([0 param.trial_duration]); 
    grid on; box off;
%     xlabel('Time (s)','FontName','Arial','FontWeight','bold');
    xticks(0:0.5:10); xticklabels(''); ylabel({'Averaged','\it\DeltaF/F_{0}'},'FontName','Arial','FontWeight','bold');
%     title({strrep([file_name, ' - ', data_name, ' - ', 'Neuron ', num2str(neuron)], '_', ' '), 'resampled dff'});
    
    nexttile(3,[1 2]);%h2 = subplot(10,4,5:8);
    imagesc(raw_calcium);
    xticks(''); ylabel({'Trial','index'},'FontName','Arial','FontWeight','bold');
    colorbar('Position',[0.14,0.55,0.02,0.1],'AxisLocation','in','FontName','Arial','Color',[1,1,1]);
%     xlabel('Time (s)','FontName','Arial','FontWeight','bold');
%     title('dff heatmap');
    
    nexttile(5,[1 2]);%h3 = subplot(10,4,9:12);
    imagesc(rate);
    xticks(''); ylabel({'Trial','index'},'FontName','Arial','FontWeight','bold');
    colorbar('Position',[0.14,0.35,0.02,0.1],'AxisLocation','in','FontName','Arial','Color',[1,1,1]);
%     xlabel('Time (s)','FontName','Arial','FontWeight','bold');
%     title('predicted spike rate');
    colormap turbo;
    
    h4=nexttile(7,[1 2]);%h4 = subplot(10,4,13:16);
    imagesc(spike);
    xticks([1,60:60:600]); xticklabels(0:10); ylabel({'Trial','index'},'FontName','Arial','FontWeight','bold');
    xlabel('Time (s)','FontName','Arial','FontWeight','bold');
%     title('predicted spike event');
    colormap(h4,[0,0,0;1,1,1]);

    set(findall(gcf,'-property','FontSize'),'FontSize',12);
    set(findall(gcf,'-property','FontName'),'FontName','Arial');

    
    if is_saving
        drawnow;
        saveas(gcf,['./saved_image/',data_name, '_ENS2_', num2str(neuron, '%03d'), '.png']);
        close all;
    end
    
    
end


function out_data = reshape_data(in_data, sampling_rate)

    global param
    out_data = reshape(in_data, param.trial_duration * sampling_rate, param.trial_num)';

end

