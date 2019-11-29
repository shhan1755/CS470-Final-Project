// colorwheel demo for Adafruit RGBmatrixPanel library.
// Renders a nice circle of hues on our 32x32 RGB LED matrix:
// http://www.adafruit.com/products/607

// Written by Limor Fried/Ladyada & Phil Burgess/PaintYourDragon
// for Adafruit Industries.
// BSD license, all text above must be included in any redistribution.

#include <RGBmatrixPanel.h>

// Most of the signal pins are configurable, but the CLK pin has some
// special constraints.  On 8-bit AVR boards it must be on PORTB...
// Pin 8 works on the Arduino Uno & compatibles (e.g. Adafruit Metro),
// Pin 11 works on the Arduino Mega.  On 32-bit SAMD boards it must be
// on the same PORT as the RGB data pins (D2-D7)...
// Pin 8 works on the Adafruit Metro M0 or Arduino Zero,
// Pin A4 works on the Adafruit Metro M4 (if using the Adafruit RGB
// Matrix Shield, cut trace between CLK pads and run a wire to A4).


#define CLK 11 // USE THIS ON ARDUINO MEGA
#define OE   9
#define LAT 10
#define A   A0
#define B   A1
#define C   A2
#define D   A3

uint8_t r;
uint8_t g;
uint8_t b;
char char_r;
char char_g;
char char_b;
char c;


RGBmatrixPanel matrix(A, B, C, D, CLK, LAT, OE, false);


void initialize();
void happy();
void sad();
void disgust();
void angry();
void surprise();
void fear();
void setup() {
  Serial.begin(9600);
  matrix.begin();
  initialize();
}

void loop() {
  // Do nothing -- image doesn't change

  while (!Serial.available());
  c = Serial.read();
  Serial.write(c);
  if (c == '0')angry();
  if (c == '1')disgust();
  if (c == '2')fear();
  if (c == '3')happy();
  if (c == '4')sad();
  if (c == '5')surprise();
}

// draw_ functions are draw image as a axial symmetry
void draw_Rect(int x,int y,int w,int h,int r,int g,int b) {
  matrix.fillRect(x,y,w,h, matrix.Color333(r,g,b));
  matrix.fillRect(32-x-w, y,w,h, matrix.Color333(r,g,b));
}

void draw_Line(int x1, int y1, int x2, int y2, int r, int g, int b) {
  matrix.drawLine(x1,y1,x2,y2, matrix.Color333(r,g,b));
  matrix.drawLine(31-x2,y2,31-x1,y1, matrix.Color333(r,g,b));
}

void draw_Rounded_Rect(int x, int y, int w, int h, int r, int g, int b) {
  draw_Rect(x,y+1,w,h-2, r,g,b);
  draw_Rect(x+1,y,w-2,h, r,g,b);
}

void eye_1(){
  draw_Rounded_Rect(5,16,6,8,0,0,0);
  draw_Rounded_Rect(6,18,4,4,2,2,2);
}

void eye_2(){
  draw_Rect(5,19,6,2,0,0,0);
}

void eye_3(){
  draw_Rect(5,19,6,2,0,0,0);
  matrix.drawRect(5,17,3,2,matrix.Color333(0,0,0));
  matrix.drawRect(21,17,3,2,matrix.Color333(0,0,0));
}
void eye_4(){
  draw_Rect(0,20,16,2,0,0,0);
  draw_Rect(1,18,13,2,0,0,0);
  draw_Rect(3,16,9,2,0,0,0);
  matrix.drawLine(8,17,10,19,matrix.Color333(2,2,2));
  matrix.drawLine(10,17,12,19,matrix.Color333(2,2,2));
  matrix.drawLine(25,17,27,19,matrix.Color333(2,2,2));
  matrix.drawLine(27,17,29,19,matrix.Color333(2,2,2));
}
void eye_5(){
  draw_Rounded_Rect(6,14,5,6,0,0,0);
  draw_Line(10,20,13,20,0,0,0);
  draw_Line(8,21,11,21,0,0,0);
  draw_Line(4,22,9,22,0,0,0);
  draw_Line(1,23,7,23,0,0,0);
}

void eye_6(){
  draw_Rect(4,17,10,4,0,0,0);
  draw_Rect(5,15,8,8,0,0,0);
  draw_Rect(7,14,4,10,0,0,0);
  draw_Rect(5,18,8,2,2,2,2);
  draw_Rect(6,16,6,6,2,2,2);
  draw_Rect(8,15,2,8,2,2,2);
  draw_Rounded_Rect(7,17,4,4,0,0,0);
}

void eye_7(){
  draw_Rounded_Rect(5,15,6,6,0,0,0);
  draw_Line(3,23,9,25,0,0,0);
}

void mouth_1(){
  matrix.fillRect(9,4,14,2, matrix.Color333(0,0,0));
}

void mouth_2() {
  draw_Rect(6,6,10,3,0,0,0);
  draw_Rounded_Rect(7,3,10,4,0,0,0);
  matrix.drawLine(10,2,21,2,matrix.Color333(0,0,0));
  draw_Rect(7,7,9,2,2,2,2);
}
void mouth_3() {
  draw_Rounded_Rect(13,3,6,4,0,0,0);
}

void mouth_4(){
  draw_Rounded_Rect(10,3,12,5,0,0,0);
}

void mouth_5() {
  draw_Rect(9,4,2,2,0,0,0);
  draw_Rect(10,5,2,2,0,0,0);
  draw_Rect(12,6,4,2,0,0,0);
}
void effect_1(){
  matrix.fillRect(3,20,2,2,matrix.Color333(1,4,1));
  matrix.fillRect(3,23,2,2,matrix.Color333(1,4,1));
  matrix.fillRect(2,25,4,5,matrix.Color333(1,4,1));
  matrix.fillRect(1,26,6,3,matrix.Color333(1,4,1));
  matrix.fillRect(3,26,2,2,matrix.Color333(4,2,0));
  matrix.fillRect(5,25,2,2,matrix.Color333(4,2,0));
}
void effect_2(){
  draw_Rounded_Rect(1,8,7,4,6,2,2);
}
void effect_3(){
  draw_Rect(5,0,6,20,2,2,3);
}
void effect_4(){
  matrix.fillRect(0,24,32,8,matrix.Color333(1,1,4));
  matrix.fillRect(0,21,32,3,matrix.Color333(1,1,3));
  matrix.fillRect(0,20,32,1,matrix.Color333(2,1,1));
}
void effect_5(){
  matrix.fillRect(1,22,5,4, matrix.Color333(2,2,3));
  matrix.fillRect(2,21,3,7, matrix.Color333(2,2,3));
  matrix.drawLine(3,28,3,30, matrix.Color333(2,2,3));
}
void effect_6(){
  matrix.fillRect(0,24,32,8,matrix.Color333(4,0,0));
  matrix.fillRect(0,21,32,3,matrix.Color333(1,0,0));
  matrix.fillRect(0,20,32,1,matrix.Color333(2,1,0));
}

void initialize() {
  matrix.fillScreen(matrix.Color333(4,2,0));
  effect_2();
  mouth_1();
  eye_1();
  delay(2800);
  draw_Rect(5,16,6,8,4,2,0);
  eye_2();
  delay(200);
  eye_1();
  
}

void happy() {
  matrix.fillScreen(matrix.Color333(4,2,0));
  effect_2();
  mouth_2();
  int i = rand()%2; // happy reaction has 2 version eye and print them randomly
  if (i==1) eye_4();
  else eye_1();
  delay(3000);
  initialize();
}

void fear() {
  matrix.fillScreen(matrix.Color333(4,2,0));
  effect_4();
  mouth_4();
  int i = rand()%2; // fear reaction has 2 version eye and print them randomly
  if (i==1) eye_7();
  else eye_1();
  delay(6000);
  initialize();
}

void sad() {
  matrix.fillScreen(matrix.Color333(4,2,0));
  eye_7();
  mouth_4();
  effect_5();

  delay(3000);
  matrix.fillScreen(matrix.Color333(4,2,0));
  effect_3();
  eye_2();
  mouth_5();
  delay(3000);
  initialize();
}

void disgust() {
  matrix.fillScreen(matrix.Color333(4,2,0));
  effect_6();
  eye_5();
  mouth_5();
  delay(1000);
  initialize();
}

void angry() {
  matrix.fillScreen(matrix.Color333(4,2,0));
  eye_7();
  mouth_4();
  effect_5();
  delay(4000);
  initialize();
}

void surprise() {
  matrix.fillScreen(matrix.Color333(4,2,0));
  eye_6();
  mouth_4();

  delay(1000);
  matrix.fillScreen(matrix.Color333(4,2,0));
  eye_7();
  mouth_4();
  effect_1();  

  delay(2000);
  initialize();
}
