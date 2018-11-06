function JoyTest
%JOYTEST Example which shows to usage of JOYMEX.
%   JOYTEST Initializes two gamepad devices and shows their six axes values
%   and their first eight button states. When only one device is connected
%   the one device will control both the left and right plots. In the
%   example certain assumptions are made about how axis might be inverted,
%   see comments in the code.
    
    running = 1;
    
    % Initialize two joysticks
    JoyMEX('init',0);
    JoyMEX('init',1);
   
    % Create figure and attach close function
    figure('CloseRequestFcn',@onClose);
    % Create plots
    subplot(2,2,1)
    p1=plot([0],[0],'x'); hold on;
    p2=plot([0],[0],'+r');
    p3=plot([0],[0],'og');
    title(sprintf('Device 0\nAxis'))
    set(gca,'xlim',[-1 1],'ylim',[-1 1]); axis square
    
    subplot(2,2,2)
    p4=plot([0],[0],'x'); hold on;
    p5=plot([0],[0],'+r');
    p6=plot([0],[0],'og');
    title(sprintf('Device 1\nAxis'))
    set(gca,'xlim',[-1 1],'ylim',[-1 1]); axis square

    subplot(2,2,3)
    b1=bar(zeros(1,8));
    title('Button States')
    set(gca,'xlim',[0 9],'ylim',[0 1]); axis square
    
    subplot(2,2,4)
    b2=bar(zeros(1,8));
    title('Button States')
    set(gca,'xlim',[0 9],'ylim',[0 1]); axis square
    
    while(running)
        % Query postion and button state of joystick 1
        [a ab] = JoyMEX(0);
        % Query postion and button state of joystick 2
        [b bb] = JoyMEX(1);
        
        % Update the plots
        
        % Notice the usage of minus signs to invert certain axis values.
        % Depending on the device you may want to change these 
        % (Left plots originally configured for Xbox 360 Controller)
        set(p1,'Xdata',a(1),'Ydata',-a(2));
        set(p2,'Xdata',a(4),'Ydata',-a(5));        
        set(p3,'Xdata',0,'Ydata',a(3));        
        set(b1,'Ydata',ab(1:8));
        % (Right plots Original configured for XGear Controller)
        set(p4,'Xdata',b(1),'Ydata',-b(2));
        set(p5,'Xdata',b(4),'Ydata',-b(5));        
        set(p6,'Xdata',b(6),'Ydata',-b(3));   
        set(b2,'Ydata',bb(1:8));

        % Force update of plot
        drawnow
    end
    
    % Clear MEX-file to release joysticks
    clear JoyMEX
    
    function onClose(src,evt)
       % When user tries to close the figure, end the while loop and
       % dispose of the figure
       running = 0;
       delete(src);
    end
end