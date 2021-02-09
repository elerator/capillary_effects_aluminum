import React from 'react'
import DrawingCanvas from './DrawingCanvas'
import { Button } from 'reactstrap';

import {QWebChannel} from 'qwebchannel'

function send_message(){
  if (typeof qt != 'undefined') {
	   alert("Qt is OK!!");
  	new QWebChannel(qt.webChannelTransport, function (channel) {
          // now you retrieve your object
          workoutObject = channel.objects.TheNameOfTheObjectUsed;
      });
    }
    else {
    	alert("Qt is not defined!");
    }
  }

function App() {
	  return <div> <Button color="primary" onClick={send_message} >primary</Button>{' '} <DrawingCanvas /></div>
}

/*const App = (props) => {
  return (
		<div>
      <Card>
      <DrawingCanvas />
        <CardBody>
          <CardTitle>Card title</CardTitle>
          <CardSubtitle>Card subtitle</CardSubtitle>
          <CardText>Some quick example text to build on the card title and make up the bulk of the card's content.</CardText>
          <Button>Button</Button>
        </CardBody>
      </Card>
    </div>
  );
}*/

export default App
