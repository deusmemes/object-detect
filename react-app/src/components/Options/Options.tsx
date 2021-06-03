import React from "react";
import styles from './Options.module.scss';
import {Input, Panel, Toggle} from "rsuite";

const Options = ({}) => {
    return (
        <Panel header={'Выберите дополнительные опции'} shaded>
            <div className={styles.option}>
                <label>Введите название</label>
                <Input placeholder={'Название'} />
            </div>
            <div className={styles.option}>
                <Toggle defaultChecked/>
                <label>Построить график динамики</label>
            </div>
        </Panel>
    )
}

export default Options;