void initialize(){
  Dxl.setPosition(dx1, 511, 150);
  Dxl.setPosition(dx2, 111, 150);
  Dxl.setPosition(dx3, 600, 150);
  Dxl.setPosition(dx4, 511, 150);
  Dxl.setPosition(dx5, 411, 150);
  Dxl.setPosition(dx6, 611, 150);
  delay(3000);
}

void disgust(){
  Dxl.setPosition(dx2, 250, 300);
  Dxl.setPosition(dx3, 480, 300);
  handshake(311,411,711,611,100,500);
  handshake(311,411,711,611,100,500);
  handshake(311,411,711,611,100,500);
  handshake(311,411,711,611,100,500);
  Dxl.setPosition(dx2, 111, 300);
  Dxl.setPosition(dx3, 600, 300);
  delay(200);
  initialize();
}

void angry(){
  Dxl.setPosition(dx2, 200, 200);
  Dxl.setPosition(dx3, 530, 200);
  handshake(611,411,411,611,400,150);
  handshake(611,411,411,611,400,150);
  handshake(611,411,411,611,400,150);
  handshake(611,411,411,611,400,150);
  handshake(611,411,411,611,400,150);
  initialize();
}         

void fear(){
  Dxl.setPosition(dx5, 311, 300);
  Dxl.setPosition(dx6, 711, 300);
  int sp = 200;
  int tmp = 400;
  Dxl.setPosition(dx4, 711, sp);
  delay(tmp);
  Dxl.setPosition(dx4, 311, sp*2);
  delay(tmp*2);
  Dxl.setPosition(dx4, 711, sp*2);
  delay(tmp*2);
  Dxl.setPosition(dx4, 311, sp*2);
  delay(tmp*2);
  Dxl.setPosition(dx4, 511, sp);
  delay(tmp);
  Dxl.setPosition(dx5, 211, 300);
  Dxl.setPosition(dx6, 811, 300);
  Dxl.setPosition(dx3,500,300);
  delay(2800);
  initialize();
}

void happy(){
  Dxl.setPosition(dx3,650,100);
  handshake(311,411,711,611,250,150);
  handshake(311,411,711,611,250,150);
  handshake(311,411,711,611,250,150);
  handshake(311,411,711,611,250,150);
  delay(1000);
  initialize();
}

void sad(){
  Dxl.setPosition(dx5, 311, 300);
  Dxl.setPosition(dx6, 711, 300);
  delay(3000);
  Dxl.setPosition(dx5, 511, 300);
  Dxl.setPosition(dx6, 511, 300);
  delay(400);
  handshake(411,511,611,511,400,150);
  handshake(411,511,611,511,400,150);
  handshake(411,511,611,511,400,150);
  Dxl.setPosition(dx5, 411, 150);
  Dxl.setPosition(dx6, 611, 150);
  delay(200);
  initialize();
}

void surprise(){
  Dxl.setPosition(dx2,161,300);
  Dxl.setPosition(dx3,650,300); 
  Dxl.setPosition(dx5, 311, 300);
  Dxl.setPosition(dx6, 711, 300);
  delay(1000);
  Dxl.setPosition(dx1, 470, 150);
  Dxl.setPosition(dx2, 111, 150);
  Dxl.setPosition(dx3, 600, 150);
  Dxl.setPosition(dx5, 611, 300);
  Dxl.setPosition(dx6, 411, 300);
  delay(2000);
  initialize();
}

void handshake(int m1, int m2, int n1, int n2, int delay_t, int sp){
  Dxl.setPosition(dx5, m1, sp);
  Dxl.setPosition(dx6, n1, sp);
  delay(delay_t);
  Dxl.setPosition(dx5, m2, sp);
  Dxl.setPosition(dx6, n2, sp);
  delay(delay_t);
}




