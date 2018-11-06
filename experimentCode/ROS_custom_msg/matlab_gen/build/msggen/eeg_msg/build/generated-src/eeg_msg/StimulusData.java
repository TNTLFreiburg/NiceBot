package eeg_msg;

public interface StimulusData extends org.ros.internal.message.Message {
  static final java.lang.String _TYPE = "eeg_msg/StimulusData";
  static final java.lang.String _DEFINITION = "std_msgs/Header  header\nstd_msgs/Float64MultiArray rawSignal\nstd_msgs/Float64MultiArray joystickData\nstd_msgs/Float64MultiArray prediction\nstd_msgs/Float64MultiArray groundTruth\nstd_msgs/String electrodeLabels\nstd_msgs/String stimulusState\n";
  std_msgs.Header getHeader();
  void setHeader(std_msgs.Header value);
  std_msgs.Float64MultiArray getRawSignal();
  void setRawSignal(std_msgs.Float64MultiArray value);
  std_msgs.Float64MultiArray getJoystickData();
  void setJoystickData(std_msgs.Float64MultiArray value);
  std_msgs.Float64MultiArray getPrediction();
  void setPrediction(std_msgs.Float64MultiArray value);
  std_msgs.Float64MultiArray getGroundTruth();
  void setGroundTruth(std_msgs.Float64MultiArray value);
  std_msgs.String getElectrodeLabels();
  void setElectrodeLabels(std_msgs.String value);
  std_msgs.String getStimulusState();
  void setStimulusState(std_msgs.String value);
}
