import { useEffect, useState } from "react";
import {
  Box,
  BoxContainer,
  Button,
  Container,
  Header,
  Input,
  InputContainer,
  LeftContainer,
  RightContainer,
} from "./styles";
import Select from "react-dropdown-select";
import axios from "axios";
import austImage from "./aust logo.png";
import facImage from "./foe-removebg-preview 1.png";

function App() {
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [stockName, setStockName] = useState(null);
  const [liveData, setLiveData] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleOnClick = () =>
    axios.post("http://localhost:5000/predict", {
      startDate,
      endDate,
      stockName,
    }).then((res) => setResult(res.data))

  const getLiveData = (s, e) =>
    axios
      .get(
        `https://api.polygon.io/v2/aggs/ticker/${stockName}/range/1/day/${s}/${e}?adjusted=true&sort=asc&apiKey=N9gB1_BjxDuyHypv4EUqNbI3MnrzMmvn`
      )
      .then((res) => {
        setLiveData(res?.data?.results?.[res.data.results.length - 1]);
      });

  useEffect(()=>{
    if(startDate&&endDate&&stockName){
      getLiveData(startDate,endDate);
    }
    if(liveData){
      setIsLoading(false)
    }
  },[stockName])

  return (
    <Container>
      <LeftContainer>
        <InputContainer>
          <p style={{ color: "white" }}>Please choose stock name</p>
          <Select
            options={[
              { value: "TSLA", label: "Tesla" },
              { value: "AAPL", label: "Apple" },
            ]}
            onChange={(values) => {
              setLiveData(null);
              setStockName(values[0]?.value);
            }}
            style={{
              width: "100%",
              backgroundColor: "#262730",
              border: "none",
              color: "gray",
              borderRadius: "5px",
              height: "40px",
              paddingLeft: "20px",
              paddingRight: "20px",
            }}
            color="black"
          />
        </InputContainer>
        <InputContainer>
          <p style={{ color: "white" }}>Start date</p>
          <Input
            type="date"
            onChange={({ target: { value } }) => {
              setStartDate(value);
              if (endDate) {
                getLiveData(value, endDate);
              }
            }}
          />
        </InputContainer>
        <InputContainer>
          <p style={{ color: "white" }}>End date</p>
          <Input
            type="date"
            onChange={({ target: { value } }) => {
              setEndDate(value);
              if (startDate) {
                getLiveData(startDate, value);
              }
            }}
          />
        </InputContainer>

        <InputContainer
          style={{
            backgroundColor: "#173A27",
            borderRadius: "5px",
            marginTop: "10px",
            height: "15%",
          }}
        >
          <p style={{ marginLeft: "10px", color: "white" }}>
            Start date : {startDate}
          </p>
          <p style={{ marginLeft: "10px", color: "white" }}>
            End date : {endDate}
          </p>
        </InputContainer>
        {stockName && liveData && !isLoading && (
          <BoxContainer>
            <Box>
              <p style={{ color: "white" }}>Height</p>
              <div
                style={{
                  width: "70%",
                  height: "30%",
                  border: "1px solid white",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <p style={{ color: "white" }}>{liveData?.h}</p>
              </div>
            </Box>
            <Box>
              <p style={{ color: "white" }}>Low</p>
              <div
                style={{
                  width: "70%",
                  height: "30%",
                  border: "1px solid white",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <p style={{ color: "white" }}>{liveData?.l}</p>
              </div>
            </Box>
            <Box>
              <p style={{ color: "white" }}>Open</p>
              <div
                style={{
                  width: "70%",
                  height: "30%",
                  border: "1px solid white",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <p style={{ color: "white" }}>{liveData?.o}</p>
              </div>
            </Box>
            <Box>
              <p style={{ color: "white" }}>Close</p>
              <div
                style={{
                  width: "70%",
                  height: "30%",
                  border: "1px solid white",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <p style={{ color: "white" }}>{liveData?.c}</p>
              </div>
            </Box>
          </BoxContainer>
        )}
      </LeftContainer>
      <RightContainer>
        <Header>
          <img src={facImage} style={{ width: "160px", height: "150px" }} />
          <p
            style={{
              color: "white",
              textAlign: "center",
              fontSize: "27px",
              fontWeight: "bold",
            }}
          >
            American University of <br />
            Science and Technology
          </p>
          <img src={austImage} style={{ width: "150px", height: "150px" }} />
        </Header>
        <p
          style={{
            color: "white",
            marginLeft: "-20%",
            marginTop: "-20%",
            textAlign: "center",
            fontSize: "22px",
          }}
        >
          Faculty of Engineering <br /> Department of Computer and
          Communications Engineering <br />
          Department of Mechatronics Engineering
          <br />
        </p>
        <p style={{ color: "white", marginLeft: "-20%", fontSize: "26px" }}>
          The Adjusted Predicted Close Price:{" "}
          <span style={{ fontWeight: "bold" }}>{result}</span>
        </p>
        <Button onClick={handleOnClick}>PREDICT</Button>
      </RightContainer>
    </Container>
  );
}

export default App;
