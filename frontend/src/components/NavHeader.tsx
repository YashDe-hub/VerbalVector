import React from "react";
import type { NavView } from "../api";

interface NavHeaderProps {
  activeView: NavView;
  onNavigate: (view: NavView) => void;
}

const NavHeader: React.FC<NavHeaderProps> = ({ activeView, onNavigate }) => {
  const navItems: { label: string; view: NavView }[] = [
    { label: "Analyze", view: "analysis" },
    { label: "Ask", view: "query" },
    { label: "History", view: "history" },
  ];

  return (
    <header
      style={{
        padding: "1.5rem 2rem",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        borderBottom: "1px solid #e2e8f0",
        position: "sticky",
        top: 0,
        backgroundColor: "white",
        zIndex: 10,
        width: "100%",
        boxSizing: "border-box",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <div
          style={{
            width: "2rem",
            height: "2rem",
            borderRadius: "9999px",
            backgroundColor: "#6366f1",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <span style={{ color: "#fff", fontWeight: 600, fontSize: "1.125rem" }}>V</span>
        </div>
        <h1 style={{ fontSize: "1.5rem", fontWeight: 300, letterSpacing: "0.025em", color: "#334155" }}>
          <span style={{ fontWeight: 500 }}>Verbal</span> Vector
        </h1>
      </div>
      <nav style={{ display: "flex", gap: "1.5rem", fontSize: "0.875rem", fontWeight: 500 }}>
        {navItems.map((item) => (
          <a
            key={item.view}
            href="#"
            onClick={(e) => {
              e.preventDefault();
              onNavigate(item.view);
            }}
            style={{
              color: activeView === item.view ? "#6366f1" : "#475569",
              textDecoration: "none",
              borderBottom: activeView === item.view ? "2px solid #6366f1" : "2px solid transparent",
              paddingBottom: "0.25rem",
              transition: "color 0.2s ease-in-out",
            }}
          >
            {item.label}
          </a>
        ))}
      </nav>
    </header>
  );
};

export default NavHeader;
