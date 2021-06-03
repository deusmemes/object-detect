import React from 'react';
import styles from './Results.module.scss';
import {ResultData} from "../../common/types";
import {Panel} from "rsuite";
import {CartesianGrid, Line, LineChart, Tooltip, XAxis, YAxis} from "recharts";

type ResultsProps = {
    data?: ResultData
}

const Results = ({data}: ResultsProps) => {
    return (
        <div>
            <Panel header={'Результаты'} shaded>
                {data
                    ? data.images.map((img, k) => <img style={{ margin: '1em' }} src={img} key={k}/>)
                    : <p>Здесь будут показаны результаты анализа</p>
                }
            </Panel>
            {
                data &&
                <Panel shaded style={{ marginTop: '3em' }} header={'Динамика изменения'}>
                    <LineChart width={700} height={300} data={data?.areas.map((area, i) => {
                        return {
                            number: i + 1,
                            area: area.toFixed(2)
                        }
                    })}>
                        <XAxis dataKey="number" label={'Номер'}/>
                        <YAxis label={'Изменение'}/>
                        <CartesianGrid stroke="#eee" strokeDasharray="5 5"/>
                        <Line type="monotone" dataKey="area" stroke="#8884d8"/>
                        <Tooltip />
                    </LineChart>
                </Panel>
            }
        </div>
    )
}

export default Results;