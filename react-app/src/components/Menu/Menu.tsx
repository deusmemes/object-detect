import React from "react";
import {Icon, Nav, Navbar} from "rsuite";

const Menu = ({}) => {
    return (
        <Navbar appearance={'inverse'}>
            <Navbar.Body>
                <Nav>
                    <Nav.Item icon={<Icon icon="home"/>}>Главная</Nav.Item>
                    <Nav.Item>Примеры работы</Nav.Item>
                    <Nav.Item>Для разработчиков</Nav.Item>
                </Nav>
                <Nav pullRight>
                    <Nav.Item icon={<Icon icon="cog" />}>Настройки</Nav.Item>
                </Nav>
            </Navbar.Body>
        </Navbar>
    )
}

export default Menu;