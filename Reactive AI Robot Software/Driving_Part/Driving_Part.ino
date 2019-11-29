#define DXL_BUS_SERIAL3 3  //Dynamixel on Serial3(USART3)  <-OpenCM 485EXP
/* Dynamixel ID defines */

#define dx1 1 // head
#define dx2 2 // neck1
#define dx3 3 // neck2
#define dx4 4 // neck3
#define dx5 5 // arm1
#define dx6 6 // arm2
/* Control table defines */
#define GOAL_POSITION 30

Dynamixel Dxl(DXL_BUS_SERIAL3);

void initialize();
int op_num = 0;

void setup() {
  SerialUSB.begin();

  Dxl.begin(3);
  Dxl.jointMode(dx1); //jointMode() is to use position mode
  Dxl.jointMode(dx2);
  Dxl.jointMode(dx3);
  Dxl.jointMode(dx4);
  Dxl.jointMode(dx5);
  Dxl.jointMode(dx6);

  initialize();
}


void loop() {  
  if(SerialUSB.available()){
    op_num = SerialUSB.read();
    SerialUSB.println(op_num);

    switch (op_num){
    case 48:
      angry();
      break;
    case 49:
      disgust();
      break;
    case 50:
      fear();
      break;
    case 51:
      happy();
      break;
    case 52:
      sad();
      break;
    case 53:
      surprise();
      break;
    defalut:
      initialize();
      break;
    }
  }
}





