%% Main script for the analysis of the experiments for the "Be Nice, Bot" paper.
%
% dependencies: 
%   Modi1987/KST-Kuka-Sunrise-Toolbox
%       (https://github.com/Modi1987/KST-Kuka-Sunrise-Toolbox)
%   V-REP
%       (http://www.coppeliarobotics.com/)
%   brewermap
%       (https://de.mathworks.com/matlabcentral/fileexchange/45208-colorbrewer-attractive-and-distinctive-colormaps)
%
% 2018, Martin Völker

%% Instructions
% First, load data file of respective subject, e.g.:
load('allFiles_moderateExp.mat');

%then, run the respective sections within the script individually
%% add path to kuka sunrise toolbox

addpath('.\Modi1987-KST-Kuka-Sunrise-Toolbox-7b72637\realTimeControl_iiwa_From_Vrep');
%% Simulation in V-Rep (V-REP scene has to be opened and expecting a connection)

% get score
x = JoystickData(:,4);
y = JoystickData(:,5);

score = (abs(atan2d(y,x))-90) / 90;



% Declare V-rep objects

vrep=remApi('remoteApi');
vrep.simxFinish(-1); 
clientID=vrep.simxStart('127.0.0.1',19999,true,true,5000,5);


if (clientID>-1)  
   
    % Get Joint handles
    jHandles=zeros(7,1);
    for i=1:7
            s=['LBR_iiwa_7_R800_joint',num2str(i)];
            [res, daHandle]=vrep.simxGetObjectHandle(clientID,s,vrep.simx_opmode_oneshot_wait);
            jHandles(i)=daHandle;
    end
    
    % gripper handle
    s='BarrettHand_jointA_0';
    [~, gripHandle]=vrep.simxGetObjectHandle(clientID,s,vrep.simx_opmode_oneshot_wait);
    
    % Finger tip sensor 0 handle
    s='BarrettHand_finderTipSensor0';
    [~, FingerTipHandle0]=vrep.simxGetObjectHandle(clientID,s,vrep.simx_opmode_oneshot_wait);
    % Finger tip sensor 1 handle
    s='BarrettHand_finderTipSensor1';
    [~, FingerTipHandle1]=vrep.simxGetObjectHandle(clientID,s,vrep.simx_opmode_oneshot_wait);
    % Finger tip sensor 2 handle
    s='BarrettHand_finderTipSensor2';
    [~, FingerTipHandle2]=vrep.simxGetObjectHandle(clientID,s,vrep.simx_opmode_oneshot_wait);
        
    % head handle
    s='Bill_head';
    [~, headHandle]=vrep.simxGetObjectHandle(clientID,s,vrep.simx_opmode_oneshot_wait);
    
    
else 
    error('The connection to V-Rep seems to have failed.');
end


% live colormap plot of rating

% figure tight formatting
iptsetpref('ImshowBorder','tight');
%removes menu and toolbar from all new figures
set(0,'DefaultFigureMenu','none');
%makes disp() calls show things without empty lines
format compact

% hf = figure('MenuBar', 'None', 'color', [1 1 1], 'Name', 'Live Rating', 'Numbertitle', 'off');
hold on;
set(gca, 'clim', [-1,1])
hcb = colorbar;
cMap = colormap(brewermap(512,'RdBu'));
colormap(cMap);
ylabel(hcb, 'subjective rating');
set(gca, 'box','off','XTickLabel',[],'XTick',[],'YTickLabel',[],'YTick',[])

vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait);
for i=1:7
        [res,tempPos]=vrep.simxGetJointPosition(clientID,jHandles(i),...
            vrep.simx_opmode_streaming);
end

jointPauseTimes = [diff(JointStateTime)/100, 0];
timeFactor = 10; % make simulation faster

handBasePosition = nan(nJointStateData,3);
effectorDistance = nan(nJointStateData,1);

for iTime = 1:nJointStateData
    
    for i=1:7
        vrep.simxSetJointPosition(clientID,jHandles(i), JointStateData_Position(iTime,i), vrep.simx_opmode_streaming);
    end
    
     pause(0.001); %comment if a live rating plot is not desired (slows sim down)
     
    %get hand base position relative to head
    [~, handBasePosition(iTime,:)] =  vrep.simxGetObjectPosition(clientID, gripHandle, headHandle, vrep.simx_opmode_oneshot_wait);
    
     imagesc(score(iTime)); %comment if a live rating plot is not desired (slows sim down)

end


vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait);


%figure settings back to default
set(0,'Default');
%%

effectorDistance = sqrt(handBasePosition(:,1).^2 + handBasePosition(:,2).^2 + handBasePosition(:,3).^2);

bagName = '';
save([bagName '_sim.mat'], 'handBasePosition', 'effectorDistance', 'score', 'bagName');

%% get joint angular velocities
disp(['The mean Score of this round was ' num2str(mean(score))]);
StimulusTime = StimulusTime-StimulusTime(1);

JointStateTime = JointStateTime-JointStateTime(1);
%
JointStateData_Position_diff = diff(JointStateData_Position);
JointStateTime_diff = diff(JointStateTime);
JointStateTime_diff_rep = repmat(JointStateTime_diff, 7,1)';

JointStateData_Position_velocity = JointStateData_Position_diff ./ JointStateTime_diff_rep;


effectorDistance_diff = diff(effectorDistance);


effector_velocity = effectorDistance_diff ./ JointStateTime_diff';


%% Plot some info

close gcf;

figure('color', 'white', 'units', 'normalized', 'position', [0,0,1,1]);

subplot(2,2,1);
plot(StimulusTime,score,'.');
hold on;
plot(StimulusTime, effectorDistance, 'r.');

title('score')
xlabel('time(s)')
% ylim([-pi/2, pi/2])

subplot(2,2,2);

plot(JoystickData(:,4), JoystickData(:,5),'.')
title('x/y axis')
ylim([-1, 1])


% Correlate Joint Angular Velocities with Score
for iJoint = 1:7
    [r(iJoint), p(iJoint)] = corr(JointStateData_Position_velocity(:,iJoint), score(2:end), 'type', 'Spearman');
end

subplot(2,2,3);
plot(r);
title('correlation - angular velocities vs. score');
ylim([-1, 1])


% Correlate end-effector distance with Score
figure;
[r_distance, p_distance] = corr(effectorDistance, score, 'type', 'Pearson');
%plot(effectorDistance, score,'.')
% Correlate end-effector velocity with Score
[r_velocity, p_velocity] = corr(effector_velocity, score(2:end), 'type', 'Pearson');
plot(effector_velocity, score(2:end),'.')

%% plot score and positions
figure('color', 'white', 'units', 'normalized', 'Position', [0.1, 0.1, 0.5, 0.5]);
set(gca, 'clim', [-1,1])
hc = colorbar;
% cMap = colormap(brewermap(512,'RdBu'));
cMap = colormap(brewermap(512,'RDYlBu'));
colormap(cMap);

hold on;

for iTime =  1:nStimulusData
    plot3(handBasePosition(iTime,1), handBasePosition(iTime,2), handBasePosition(iTime,3),...
        '.', 'color', cMap(ceil((score(iTime)+1.001)*256),:))
%     pause (0.00001)
end

% Head
scatter3(0,0,0, 5000, [0 0 0],'filled')
alpha(0.5)
% plot3(0.1,0.1,0.1, 'k>', 'MarkerSize',  40, 'MarkerFaceColor', [0 0 0], 'MarkerEdgeColor', [0 0 0])

xlabel('x');
ylabel('y');
zlabel('z');

grid on

set(gca, 'FontSize', 16)
set(gcf,'Units','Inches');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

%% viewing angles to choose from
az = -80.8000; el =  5.2000; % view from diagonal behind subject
view(az,el);
%
print(gcf,'view_diagonal_behind','-dpng','-r1200')

az = -180; el =  0;  % view from left side of subject
view(az,el);
print(gcf,'view_left','-dpng','-r1200')

az = -90; el =  0.0000;  % view from straight behind side of subject
view(az,el);
print(gcf,'view_straight_behind','-dpng','-r1200')

az = -90; el =  90;  % topview
view(az,el);
print(gcf,'view_top','-dpng','-r1200')


%[az,el] = view % get current viewing angle
%% Define Header

H.sf = 512;
H.cn = {'POz', 'P1', 'Fp1', 'CPz', 'CP1', 'CP3', 'C1', 'C3', 'Pz', 'P2', 'Fp2', 'CP2', 'CP4', 'Cz', 'C2', 'C4' ...
    'FCz', 'FC1', 'AF7', 'F1', 'F3', 'AFz', 'AF3', 'F5', 'FC2', 'AF8', 'Fz', 'F2', 'F4', 'AF4', 'Fpz', 'F6' ...
    'ECG1', 'ECG2', 'breathingBelt1', 'breathingBelt2'};

%% Preprocess EEG
EEGData(:,33) = EEGData(:,34) - EEGData(:,33); %re-reference ECG
H.cn{33} = 'ECG';
EEGData(:,35) = EEGData(:,36) - EEGData(:,35); %re-reference Respiration
H.cn{35} = 'Respiration';

EEGData(:,36) = [];
H.cn(36) = []; 
EEGData(:,34) = [];
H.cn(34) = []; 


%% Get score development at specific positions

az = -80.8000; el =  5.2000; % view from diagonal behind subject
view(az,el);

az = -180; el =  0;  % view from left side of subject
view(az,el);

az = -90; el =  0.0000;  % view from straight behind side of subject
view(az,el);

az = -90; el =  90;  % topview
view(az,el);




% +- 0.2
pos{1} = [1.032, -0.48, 0.689]; %_init
pos{2} = [1.429, -0.655, -0.307]; %_grasping
pos{3} = [0.099, -0.70, 0.509]; %_overhead
pos{4} = [0.518, 0.038, -0.205]; %_correct_end
pos{5} = [0.297, -0.30, 0.44]; %_wrong_end

%%
distance = 0.01; % distance from specific position to count as position data
clear score_pos score_pos_movAvg score_pos_resample


for iPos = 1:numel(pos)
    score_pos{iPos} = score; %score
    score_pos{iPos}(handBasePosition(:,1) < (pos{iPos}(1)-distance)) =  nan;
    score_pos{iPos}(handBasePosition(:,1) > (pos{iPos}(1)+distance)) =  nan;
    
    score_pos{iPos}(handBasePosition(:,2) < (pos{iPos}(2)-distance)) =  nan;
    score_pos{iPos}(handBasePosition(:,2) > (pos{iPos}(2)+distance)) =  nan;
    
    score_pos{iPos}(handBasePosition(:,3) < (pos{iPos}(3)-distance)) =  nan;
    score_pos{iPos}(handBasePosition(:,3) > (pos{iPos}(3)+distance)) =  nan;
    
    JointStateData_Position_copy{iPos} = JointStateData_Position;
    JointStateData_Position_copy{iPos}(isnan(score_pos{iPos}),:) = [];
    score_pos{iPos}(isnan(score_pos{iPos})) = [];
    
    
    mean_score(iPos) = mean(score_pos{iPos});
    std_score(iPos) = std(score_pos{iPos});
    SEM_score(iPos) = std_score(iPos)/numel(score_pos{iPos});
end

% moveavg
WindowType = 'gauss';
% span = 2; %in s
% span = 16*span; %in score samples (score is in 16 Hz)

nResample = 50;

for iPos = 1:numel(pos)
    span = round(numel(score_pos{iPos})/5);
    
    if strcmpi(WindowType, 'rect')
        mask = ones(span, 1) / span ; %rectangular window
    elseif strcmpi(WindowType, 'gauss')
        mask = gausswin(span); mask = mask/sum(mask);    % gaussian weighted window
    end

    
    score_pos_movAvg{iPos} = convn(score_pos{iPos}, mask, 'valid');
 
    score_pos_resample{iPos} = downsample(score_pos_movAvg{iPos}, floor(numel(score_pos_movAvg{iPos})/nResample));
    
       
end

% define colors for plotting
colors{1} = [];%_init
colors{2} = [0 0 1]; %_grasping
colors{3} = [255,215,0]/255; %_overhead
colors{4} = [34,139,34]/255;%_correct_end
colors{5} = [1 0 0]; %_wrong_end

figure('color', 'white', 'units', 'normalized', 'position', [0.3,0.3,0.8,0.5]);
hold on;
for iPos = 2:numel(pos)
    plot(score_pos_resample{iPos}, 'linewidth', 3, 'color', colors{iPos});
end
line([0,nResample], [0 0], 'Linestyle',':', 'Linewidth', 2, 'color', [0 0 0]);
line([0,nResample], [0.5 0.5], 'Linestyle',':', 'Linewidth', 1, 'color', [0 0 0]);
line([0,nResample], [-0.5 -0.5], 'Linestyle',':', 'Linewidth', 1, 'color', [0 0 0]);
line([0,nResample], [1 1], 'Linestyle','-', 'Linewidth', 0.5, 'color', [0 0 0]);
line([0,nResample], [-1 -1], 'Linestyle','-', 'Linewidth', 0.5, 'color', [0 0 0]);
line([0,0], [-1 1], 'Linestyle','-', 'Linewidth', 0.5, 'color', [0 0 0]);
line([nResample,nResample], [-1 1], 'Linestyle','-', 'Linewidth', 0.5, 'color', [0 0 0]);
legend({'grasping', 'over-head', 'correct end', 'wrong end'}, 'FontSize', 24);
set(gca, 'FontSize', 24);

ylim([-1,1]);
xlim([1, nResample]);
ylabel('subjective rating');
set(gca,'xtick', [1,nResample], 'xticklabel', {'start', 'end'});
xlabel('normalized experiment time');

set(gcf,'PaperPositionMode', 'Auto');

title('no robot experience')
%% save fig
print(gcf,'score_development_no_exp','-dpng','-r300');

%% convert to BBCI format
Settings.convertForConvNet = 1;
H.PatientSession = 'modExp'; %noExp, substExp

if Settings.convertForConvNet
    
    nChannels = numel(H.cn);
    
    % also save hand position
    handBasePosition_x_resample =   resample(score, size(D.data,2), size(handBasePosition(:,1),1))';
    handBasePosition_y_resample =   resample(score, size(D.data,2), size(handBasePosition(:,2),1))';
    handBasePosition_z_resample =   resample(score, size(D.data,2), size(handBasePosition(:,3),1))';
    H.cn{nChannels+1} = 'robotHandPos_x';
    D.data(nChannels+1,:) = handBasePosition_x_resample;
    H.cn{nChannels+2} = 'robotHandPos_y';
    D.data(nChannels+2,:) = handBasePosition_y_resample;
    H.cn{nChannels+3} = 'robotHandPos_z';
    D.data(nChannels+3,:) = handBasePosition_z_resample;
    
    disp('------ Converting to BBCI format ------');
    overwrite = 1;
    write = 1;
    savePath = '\BBCI_files';
    saveName = [H.PatientSession '_' num2str(H.sf) 'Hz_CAR'];
    [hdr, mrk, mnt, nfo, dat] = ERN_BBCI_import(H,M,D, overwrite, write, savePath, saveName);
    
  
    % interpolate score manually to avoid filter artifacts
    for iScore = 1:numel(score)
        score_resample(iScore*16-15:iScore*16) = score(iScore);
    end
    
    
    save([savePath '\' H.PatientSession '_score.mat'], 'score_resample');

    
    
end



