import styled from "styled-components";

export const Container = styled.div`
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;
  background-color: black;
`;

export const Input = styled.input`
  background-color: #262730;
  width: calc(100% - 40px);
  color: gray;
  height: 50px;
  border-radius: 5px;
  border: none;
  padding-left: 20px;
  padding-right: 20px;
`;

export const InputContainer = styled.div`
  width: 60%;
  height: 13%;
  display: flex;
  flex-direction: column;
`;

export const BoxContainer = styled.div`
  width: 65%;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  height: 15%;
`;

export const Box = styled.div`
  width: 25%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

export const LeftContainer = styled.div`
  width: 55%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
`;
export const RightContainer = styled.div`
  width: 45%;
  height: 90vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
`;

export const Header = styled.div`
  display: flex;
  flex-direction: row;
  width: 100%;
  height: 30%;
  justify-content: space-between;
  align-items: flex-start;
  margin-left: -20%;
`;
export const Button = styled.button`
  color: white;
  border: 1px solid white;
  background-color: black;
  width: 90%;
  margin-left: -20%;
  height: 40px;
  border-radius: 10px;
  letter-spacing: 5px;
  cursor: pointer;
  &:hover {
    background-color: white;
    color: black;
  }
`;
