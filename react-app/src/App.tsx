import React, {useEffect, useState} from 'react';
import 'rsuite/dist/styles/rsuite-default.css';
import './App.css';
import Menu from "./components/Menu/Menu";
import SelectModel from "./components/SelectModel/SelectModel";
import UploadImage from "./components/UploadImage/UploadImage";
import {Container, Content, Footer, Header, Steps} from "rsuite";
import StepsBlock from "./components/Steps/StepsBlock";
import Options from "./components/Options/Options";
import Results from "./components/Results/Results";
import {AnalyzeData, Model, ResultData} from "./common/types";
import {HttpService} from "./services/http";
import {base64ToString, sendNotify} from "./common/utils";
import {FileType} from "rsuite/es/Uploader";
import {Message} from './services/message'

const styles = {
    width: '200px',
    display: 'inline-table',
    verticalAlign: 'top',
    position: 'absolute'
};

function App() {
    const [currentStep, setCurrentStep] = useState(0);
    const [results, setResults] = useState<ResultData | undefined>(undefined);
    const [models, setModels] = useState<string[]>([]);
    const [selectedModel, setSelectedModel] = useState('')

    const httpService = HttpService.getInstance()

    useEffect(() => {
        httpService.getModels()
            .then(res => setModels(res.data.models))
    }, []);

    const startAnalyze = (files: FileType[]) => {
        const data = {
            images: files,
            model: {id: 1, name: selectedModel, description: 'aaa'} as Model
        } as AnalyzeData

        httpService.analyze(data)
            .then(res => {
                const data = res.data as ResultData
                const results = {
                    areas: data.areas,
                    images: data.images.map(img => base64ToString(img))
                } as ResultData;
                setResults(results);
                setCurrentStep(3);
                Message.success('Есть результаты', 'Успех')
            })
            .catch(err => console.log(err))
    }

    return (
        <Container>
            <Header><Menu/></Header>
            <StepsBlock step={currentStep}/>
            <Content style={{width: '60%', marginLeft: '20%'}}>
                <Container style={{marginTop: '3em'}}>
                    <SelectModel models={models} selected={selectedModel}
                                 setSelectedModel={setSelectedModel}
                                 onChange={() => {
                                     setCurrentStep(1);
                                 }}
                    />
                </Container>
                <Container style={{marginTop: '3em'}}>
                    <UploadImage onSubmit={(files: FileType[]) => {
                        setCurrentStep(2);
                        startAnalyze(files);
                    }}/>
                </Container>
                <Container style={{marginTop: '3em'}}>
                    <Results data={results}/>
                </Container>
            </Content>
            <Footer></Footer>
        </Container>
    )
}

export default App;
