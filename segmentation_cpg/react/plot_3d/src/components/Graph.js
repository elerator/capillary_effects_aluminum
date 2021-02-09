import React from "react";
import Plotly from 'plotly.js-dist'
import createPlotlyComponent from 'react-plotlyjs';

const Plot = createPlotlyComponent(Plotly);

class Graph extends React.Component {
  constructor(props) {
    super(props);
    this.state = {width: 0,
                  height: 0,
                  graphMetaData : {
                                      title: props.title,
                                      xAxis: "X-Axis",
                                      yAxis: "Y-Axis",
                                      zAxis: "Z-Axis"

                                  },
                  graphData:{color:"blue",opcacity:1.0,x:[],y:[],z:[],i:[],j:[],k:[]}
            };
    this.updateWindowDimensions = this.updateWindowDimensions.bind(this);
    window["Graph_"+props.graph_id] = this;
  }

  componentDidMount() {
    this.updateWindowDimensions();
    window.addEventListener('resize', this.updateWindowDimensions);
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.updateWindowDimensions);
  }

  updateWindowDimensions() {
    this.setState({ width: window.innerWidth, height: window.innerHeight });
  }

  set_mesh(data){
    console.log("received");
    data = JSON.parse(data);
    console.log("parsed");
    //var graphData = this.state.graphData;
    //graphData["x"] = xyz["x"];
    //graphData["y"] = xyz["y"];
    //graphData["z"] = xyz["z"];
    this.setState({graphData : data});
    console.log("state updated")
	}

  render() {
    let meta = this.state.graphMetaData
    let data = this.state.graphData;

    return (
      <Plot
        data={[
          {
            type: "mesh3d",
            x:  data.x,
            y:  data.y,
            z:  data.z,
            i:  data.i,
            j:  data.j,
            k:  data.k,
          }
        ]}
        layout={{
          width: this.state.width,
          height: this.state.height,
          margin: {
            l: 50,
            r: 50,
            b: 80,
            t: 90,
            pad: 4
          },
          title: meta.title,
          scene: {
            xaxis: {
              title: meta.xAxis,
              titlefont: {
                family: "Courier New, monospace",
                size: 12,
                color: "#444444"
              }
            },
            yaxis: {
              title: meta.yAxis,
              titlefont: {
                family: "Courier New, monospace",
                size: 12,
                color: "#444444"
              }
            },
            zaxis: {
              title: meta.zAxis,
              titlefont: {
                family: "Courier New, monospace",
                size: 12,
                color: "#444444"
              }
            }
          }
        }}
      />
    );
  }
}

export default Graph;
