function out_signal = bci_Process( in_signal )

% Apply a filter to in_signal, and return the result in out_signal.
% Signal dimensions are ( channels x samples ).
% This file is part of BCI2000, a platform for real-time bio-signal research.
% [ Copyright (C) 2000-2012: BCI2000 team and many external contributors ]


% Parameters and states are global variables.
global bci_Parameters bci_States;
global carFilter eegpub spatialFilter dataSize nChannels nSamples bpf iCz eegmsg sub Sequence;

%apply common average re-referencing only to EEG channels
in_signal(1:32,:) = carFilter * in_signal(1:32,:);


%apply spatial filter to data
% rawData = in_signal(spatialFilter, :);



%% bandpass data
[filtData, bpf.filtConds] = filter(bpf.b, bpf.a, in_signal, bpf.filtConds, bpf.dim);

%% re-reference to Cz
% filtData = bsxfun(@minus, filtData, filtData(iCz,:));


%% send data ROS
%reshape data
data_reshape = double(reshape(filtData, dataSize(1)*dataSize(2),1));


eegmsg.RawSignal.Data = data_reshape;


%Joystick Data
[A, ~] = JoyMEX;
eegmsg.JoystickData.Data = double(A)';

% Header data
eegmsg.Header.Stamp = rostime('now');
eegmsg.Header.Seq = Sequence;

send(eegpub,eegmsg)

Sequence = Sequence+1;

%% Receive robot info from subscription, save it into states

scan  = sub.LatestMessage;


bci_States.iiwaJointPos1 = scan.Position(1);
bci_States.iiwaJointPos2 = scan.Position(2);
bci_States.iiwaJointPos3 = scan.Position(3);
bci_States.iiwaJointPos4 = scan.Position(4);
bci_States.iiwaJointPos5 = scan.Position(5);
bci_States.iiwaJointPos6 = scan.Position(6);
bci_States.iiwaJointPos7 = scan.Position(7);

bci_States.JoystickData1x = A(1);
bci_States.JoystickData1y = A(2);

bci_States.JoystickData2x = A(4);
bci_States.JoystickData2y = A(5);


%%

out_signal = zeros(nChannels, nSamples);

end