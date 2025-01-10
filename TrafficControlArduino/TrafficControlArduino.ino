int orange=13;
int red = 9;
int green1=10;
int green2=11;
int green3=12;

void setup() 
{
  pinMode(orange,OUTPUT);
  pinMode(red,OUTPUT);
  pinMode(green1,OUTPUT);
  pinMode(green2,OUTPUT);
  pinMode(green3,OUTPUT);
  Serial.begin(9600);

}

void loop() 
{
  unsigned char cmd;
  if(Serial.available()>0)
  {
    cmd = Serial.read();
  }

  if(cmd == 'O')
  {
    digitalWrite(orange,HIGH);
    digitalWrite(red,LOW);
    digitalWrite(green1,LOW);
    digitalWrite(green2,LOW);
    digitalWrite(green3,LOW);
  }
  else if(cmd == 'R')
  {
    digitalWrite(orange,LOW);
    digitalWrite(red,HIGH);
    digitalWrite(green1,LOW);
    digitalWrite(green2,LOW);
    digitalWrite(green3,LOW);    
  }
  else if(cmd == 'G')
  {
    digitalWrite(orange,LOW);
    digitalWrite(red,LOW);
    digitalWrite(green1,HIGH);
    digitalWrite(green2,HIGH);
    digitalWrite(green3,HIGH);    
  }
  else if(cmd == '1')
  {
    digitalWrite(green1,LOW);
  }
  else if(cmd == '2')
  {
    digitalWrite(green2,LOW);
  } 
  else if(cmd == '3')
  {
    digitalWrite(green3,LOW);
  } 

}
