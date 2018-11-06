@cls & "\BCI2000\prog\BCI2000Shell" %0 %* #! && exit /b 0 || exit /b 1\n \n
Change directory $BCI2000LAUNCHDIR
Show window; Set title ${Extract file base $0}
Startup system localhost
Start executable gUSBampSource --local
Start executable MatlabSignalProcessing  --local --MatlabWD="\matlabOnlineScripts"
Start executable DummyApplication --local
Wait for Connected
Load parameterfile "\NiceBot_params_gUSBamp.prm"
