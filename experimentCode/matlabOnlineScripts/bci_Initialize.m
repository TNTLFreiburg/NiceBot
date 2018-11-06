function bci_Initialize( in_signal_dims, out_signal_dims )

% Filter initialize demo
% 
% Perform configuration for the bci_Process script.

% BCI2000 filter interface for Matlab
% juergen.mellinger@uni-tuebingen.de, 2005
% $BEGIN_BCI2000_LICENSE$
% 
% This file is part of BCI2000, a platform for real-time bio-signal research.
% [ Copyright (C) 2000-2012: BCI2000 team and many external contributors ]
% 
% BCI2000 is free software: you can redistribute it and/or modify it under the
% terms of the GNU General Public License as published by the Free Software
% Foundation, either version 3 of the License, or (at your option) any later
% version.
% 
% BCI2000 is distributed in the hope that it will be useful, but
%                         WITHOUT ANY WARRANTY
% - without even the implied warranty of MERCHANTABILITY or FITNESS FOR
% A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License along with
% this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% $END_BCI2000_LICENSE$

% Add needed paths

addpath('C:\matlab_offline_toolboxes\matlab_scripts_agball')

% Parameters and states are global variables.
global bci_Parameters bci_States;
global carFilter eegpub spatialFilter dataSize nChannels nSamples bpf iCz eegmsg sub Sequence;

%% Collect information set to send to the decoder

SRnumIdcs = regexp(bci_Parameters.SamplingRate{1}, '\d'); % sometimes, there is the string 'Hz' within SR


SR =(str2double(bci_Parameters.SamplingRate{1}(SRnumIdcs)));
nChannels = str2double(bci_Parameters.SourceCh{1});
channelNames = bci_Parameters.ChannelNames;
nSamples = str2double(bci_Parameters.SampleBlockSize{1});


%% Define variables for the common average reference (car) filter and spatial filter
nEEGChannels = size(channelNames,1);
carFilter = [];
carFilter = eye(nEEGChannels) - ones(nEEGChannels)/nEEGChannels;


%% channel information

% channel names
chanString = channelNames{1};
for i = 2:numel(channelNames)
    chanString = [chanString ' ' channelNames{i}];
end

% data size
dataSize = [nChannels, nSamples];


% %% design bandpass filter
% [bpf.b,bpf.a] = butter(3, [0.5 40]/(SR/2), 'bandpass');
% bpf.filtConds = [];
% bpf.dim = 2;

iCz = find(strcmp(channelNames, 'Cz'));

%% Initialize Joystick
try
    JoyMEX('init',0);
catch
    warning('JoyMex init failed: Joystick already initalized or not connected?');
end
%% ROS node

% robot ip adress and rosmaster has to be used rosinit('master_host')
rosMasterURI =  'http://192.168.42.66:11311';
ROS_IP = '192.168.42.37';

setenv('ROS_MASTER_URI', rosMasterURI)
setenv('ROS_IP', ROS_IP)
rosinit


% the costum message type has to be created and subtitute eeg_msgs
MessageType = 'eeg_msg/StimulusData' ;
% MessageType = 'std_msgs/String';
eegpub = rospublisher('/eeg/stimulusdata', MessageType);

% fill the message content
eegmsg = rosmessage(eegpub);

eegmsg.ElectrodeLabels.Data = chanString;
% eegmsg.RawSignal.Layout.Dim.Size = dataSize;

Sequence = 1;
%% Ros subscription
sub = rossubscriber('/joint_states_iiwa', 'sensor_msgs/JointState');



